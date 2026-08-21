"""regressions for persisted-spec decoding and digest replay.

these tests bind the boundary that authored config intentionally does not share: immutable records
written by an older flash remain readable, while unknown keys outside the explicit historical
registries still fail closed.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import pytest

import flash.runner as runner
from flash.core.spec import JobSpec
from flash.core.spec_persistence import (
    DROPPED_TOP_LEVEL_KEYS,
    REMOVED_PERSISTED_TRAIN_KEYS,
)

_MANAGED_DIGEST_KEYS = (
    "model_revision_auto",
    "model_revision_force_pin",
    "gpu_count_auto",
    "workload_profile_input_digest",
    "workload_profile_producer_version",
    "workload_profile",
)
_REMOVED_TOP_LEVEL_VALUES = {
    "model_policy": "allow",
    "model_revision": "",
    "worker_env": {"OLD_OVERRIDE": "1"},
    "workload_profile_kind": "sft",
}


def _rollout_spec() -> JobSpec:
    return replace(
        JobSpec.from_dict(
            {
                "model": "Qwen/Qwen3.5-4B",
                "algorithm": "grpo",
                "run_id": "persisted-envelope",
                "project": "11111111-1111-4111-8111-111111111111",
                "environment": {},
                "gpu": {"type": "H100", "count": 1},
                "train": {"epochs": 1, "group_size": 4, "prompts_per_step": 32},
            }
        ),
        model_revision="a" * 40,
        model_revision_auto=True,
    )


def _normalized_digest_payloads(spec: JobSpec) -> tuple[dict, dict]:
    public = spec.to_dict()
    worker = spec.to_internal_dict()
    if not public["environment"].get("pip"):
        public["environment"].pop("pip", None)
    for key in _MANAGED_DIGEST_KEYS:
        if not worker.get(key):
            worker.pop(key, None)
    return public, worker


def _legacy_record() -> tuple[dict, dict, str]:
    spec = _rollout_spec()
    public, worker = _normalized_digest_payloads(spec)

    for payload in (public, worker):
        payload["train"]["batch_size"] = 32
        payload["train"].pop("prompts_per_step", None)
    public["train"].pop("lora_alpha", None)

    public.update(
        {"model_revision": spec.model_revision, "worker_env": {"PUBLIC_OLD_OVERRIDE": "1"}}
    )
    worker.update(
        {
            "model_policy": "allow",
            "worker_env": {"WORKER_OLD_OVERRIDE": "1"},
            "workload_profile_kind": "sft",
        }
    )
    payload = {
        "version": 1,
        "public_spec": public,
        "worker_spec": worker,
        "adapter_identity": None,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return public, worker, hashlib.sha256(canonical.encode()).hexdigest()


def test_persisted_decoder_tolerates_only_registered_removed_top_level_keys() -> None:
    assert set(_REMOVED_TOP_LEVEL_VALUES) == set(DROPPED_TOP_LEVEL_KEYS)
    original = JobSpec()
    for key, value in _REMOVED_TOP_LEVEL_VALUES.items():
        persisted = {**original.to_internal_dict(), key: value}
        restored = JobSpec.from_dict(persisted)
        if key == "model_revision":
            assert restored.model_revision == ""
        else:
            assert not hasattr(restored, key)

    with pytest.raises(ValueError, match=r"unknown key.*not_historical"):
        JobSpec.from_dict({**original.to_internal_dict(), "not_historical": True})


def test_persisted_decoder_tolerates_only_registered_removed_train_keys() -> None:
    assert frozenset({"advantage_clip"}) == REMOVED_PERSISTED_TRAIN_KEYS
    persisted = JobSpec().to_internal_dict()
    persisted["train"]["advantage_clip"] = 1.5
    assert JobSpec.from_dict(persisted).train == JobSpec().train

    persisted["train"]["not_historical"] = 1
    with pytest.raises(ValueError, match=r"train has unknown key.*not_historical"):
        JobSpec.from_dict(persisted)


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ({"batch_size": 32}, 32),
        ({"prompts_per_step": 16}, 16),
        ({"batch_size": 32, "prompts_per_step": 16}, 16),
        # a non-positive legacy value is DISCARDED, not rejected: `minimum=1` would have refused it
        # at submission, so a persisted record carrying it is already past that gate and raising
        # here only strands recovery and teardown on a run that can no longer be resubmitted.
        ({"batch_size": 0}, None),
    ],
)
def test_persisted_rollout_batch_migrates_without_retaining_the_old_name(
    stored: dict, expected: int | None
) -> None:
    payload = _rollout_spec().to_internal_dict()
    payload["train"].update(stored)
    if "prompts_per_step" not in stored:
        payload["train"].pop("prompts_per_step", None)

    restored = JobSpec.from_dict(payload)

    assert restored.train.prompts_per_step == expected
    assert restored.train.batch_size is None


def test_versioned_envelope_reproduces_exact_historical_digest_bytes() -> None:
    raw_public, raw_worker, expected = _legacy_record()
    public = JobSpec.from_dict(raw_public)
    worker = JobSpec.from_dict(raw_worker)
    persisted = runner.VersionedPersistedSpecEnvelope.read({"worker_spec": raw_worker}, raw_public)

    assert runner._preparation_digest(public, worker, None, persisted=persisted) == expected

    tampered = deepcopy(raw_public)
    tampered["worker_env"]["PUBLIC_OLD_OVERRIDE"] = "tampered"
    tampered_envelope = runner.VersionedPersistedSpecEnvelope.read(
        {"worker_spec": raw_worker}, tampered
    )
    assert runner._preparation_digest(public, worker, None, persisted=tampered_envelope) != expected

    next_version = replace(persisted, version=2)
    assert runner._preparation_digest(public, worker, None, persisted=next_version) != expected


def test_envelope_version_is_persisted_and_validated() -> None:
    from flash.runner.submit import _effective_preparation_snapshot

    spec = _rollout_spec()
    snapshot = _effective_preparation_snapshot(spec, spec, None)

    assert snapshot["version"] == runner.VersionedPersistedSpecEnvelope.CURRENT_VERSION
    assert runner.VersionedPersistedSpecEnvelope.read(snapshot, spec.to_dict()).version == 1
    with pytest.raises(ValueError, match="unsupported persisted preparation envelope version 2"):
        runner.VersionedPersistedSpecEnvelope.read({**snapshot, "version": 2}, spec.to_dict())
    with pytest.raises(ValueError, match="version must be a positive integer"):
        runner.VersionedPersistedSpecEnvelope.read({**snapshot, "version": True}, spec.to_dict())


def test_envelope_never_replays_a_field_the_current_spec_still_defines() -> None:
    public = {"model": "Qwen/Qwen3.5-4B", "train": {}, "environment": {}}
    worker = deepcopy(public)
    persisted = runner.VersionedPersistedSpecEnvelope(
        worker_dropped_keys={"model": "forged/model", "model_policy": "allow"},
        public_dropped_keys={"model": "forged/model", "model_revision": "deadbeef"},
    )

    persisted.rewind(public, worker)

    assert public["model"] == worker["model"] == "Qwen/Qwen3.5-4B"
    assert public["model_revision"] == "deadbeef"
    assert worker["model_policy"] == "allow"


def test_legacy_record_recovers_after_worker_half_is_re_persisted() -> None:
    from flash.runner.submit import _effective_preparation_snapshot

    raw_public, raw_worker, initial_digest = _legacy_record()
    status = runner.RunStatus(
        state="running",
        run_id="persisted-envelope",
        spec=raw_public,
        effective_preparation={
            "worker_spec": raw_worker,
            "preparation_digest": initial_digest,
        },
    )

    recovered = runner.effective_spec_from_status(status)
    public = JobSpec.from_dict(raw_public)
    persisted = runner.VersionedPersistedSpecEnvelope.read(
        status.effective_preparation, raw_public, include_worker=False
    )
    status.effective_preparation = _effective_preparation_snapshot(
        public, recovered, None, persisted=persisted
    )
    assert status.effective_preparation["version"] == 1

    recovered_again = runner.effective_spec_from_status(status)

    assert recovered_again.train.prompts_per_step == 32
    assert recovered_again.train.batch_size is None


def test_historical_public_revision_remains_structurally_bound() -> None:
    raw_public, raw_worker, digest = _legacy_record()
    raw_public["model_revision"] = "b" * 40
    status = runner.RunStatus(
        state="running",
        run_id="persisted-envelope",
        spec=raw_public,
        effective_preparation={"worker_spec": raw_worker, "preparation_digest": digest},
    )

    with pytest.raises(ValueError, match="does not match the public run"):
        runner.effective_spec_from_status(status)


def test_historical_profile_record_never_advertises_an_adapter() -> None:
    worker = JobSpec.from_dict(
        {
            **JobSpec().to_internal_dict(),
            "run_id": "historical-profile",
            "workload_profile_kind": "sft",
        }
    ).to_internal_dict()
    worker["workload_profile_kind"] = "sft"
    worker["train"]["hf_repo"] = "owner/repo"
    status = runner.RunStatus(
        state="done",
        run_id="historical-profile",
        spec=JobSpec().to_dict(),
        effective_preparation={"worker_spec": worker},
    )

    assert runner._adapter_ref_for_status(status) is None
