from __future__ import annotations

import json
from dataclasses import replace

import pytest

from flash.spec import EnvironmentSpec, GpuSpec, JobSpec, TrainSpec
from flash.workload_profile import (
    SftWorkloadProfile,
    sft_profile_input_digest,
    sft_profile_input_payload,
    sft_profile_run_id,
)


def _spec() -> JobSpec:
    return JobSpec(
        model="Qwen/Qwen3.5-0.8B",
        model_revision="a" * 40,
        algorithm="sft",
        environment=EnvironmentSpec(
            id="team/example",
            resolved_sha="b" * 40,
            params={"dataset_split": "train", "private_selector": "not-persisted"},
        ),
        train=TrainSpec(
            epochs=2,
            batch_size=8,
            max_context_tokens=1024,
            max_steps=12,
            max_examples=64,
        ),
        gpu=GpuSpec(count=2),
        seed=42,
        worker_env={"FLASH_FLA_SM120": "free"},
        thinking=True,
    )


def _profile(**overrides) -> SftWorkloadProfile:
    values = {
        "input_digest": "c" * 64,
        "producer_version": "1.2.3",
        "tokenizer_revision": "d" * 64,
        "environment_id": "team/example",
        "environment_revision": "b" * 40,
        "source_examples": 80,
        "selected_examples": 64,
        "retained_examples": 60,
        "dropped_examples": 4,
        "epochs": 2,
        "max_length": 1024,
        "packing_mode": "packed",
        "architecture_mode": "pure-attention",
        "packed_blocks": 20,
        "real_tokens_per_epoch": 15_000,
        "supervised_tokens_per_epoch": 6_000,
        "padded_compute_tokens_per_epoch": 20_000,
        "authoritative_real_tokens": 45_000,
        "authoritative_supervised_tokens": 18_000,
        "authoritative_compute_tokens": 60_000,
        "realized_max_length": 980,
        "examples_per_update": 8,
        "derived_steps": 8,
        "authoritative_steps": 12,
        "packing_efficiency": 0.75,
        "sample_policy": "exact-prefix",
        "created_at": 1_780_000_000.0,
    }
    values.update(overrides)
    return SftWorkloadProfile(**values)


def test_sft_profile_round_trips_with_a_verified_content_digest() -> None:
    profile = _profile()

    restored = SftWorkloadProfile.from_dict(profile.to_dict())

    assert restored == profile
    assert restored.content_digest == profile.content_digest


def test_sft_profile_rejects_tampered_content() -> None:
    raw = _profile().to_dict()
    raw["retained_examples"] = 59
    raw["dropped_examples"] = 5

    with pytest.raises(ValueError, match="content digest"):
        SftWorkloadProfile.from_dict(raw)


def test_sft_profile_rejects_inconsistent_aggregates() -> None:
    with pytest.raises(ValueError, match="retained plus dropped"):
        _profile(dropped_examples=3)

    with pytest.raises(ValueError, match="packing_efficiency"):
        _profile(packing_efficiency=0.8)


def test_creation_time_is_provenance_and_never_moves_the_measurement() -> None:
    """The training worker compares its own recomputation against the stored profile.

    A recomputation is not a producer, so it stamps no time. If ``created_at`` reached equality or
    the content digest, that comparison would fail on every run for the one difference that is not
    a workload change, and the fail-closed gate would become a permanent outage.
    """
    stored = _profile(created_at=1_780_000_000.0)
    recomputed = _profile(created_at=0.0)

    assert recomputed == stored
    assert recomputed.content_digest == stored.content_digest
    assert stored.to_dict()["created_at"] == 1_780_000_000.0
    assert SftWorkloadProfile.from_dict(stored.to_dict()).created_at == 1_780_000_000.0


def test_content_digest_covers_every_measured_field() -> None:
    """Guards the one risk of splitting the digest off ``asdict``: a field silently uncovered.

    Only ``created_at`` may sit outside the digest. Anything else added to the schema without a
    deliberate ``compare=False`` must move the digest, or a tampered value would round-trip.
    """
    base = _profile()
    provenance = {"created_at"}

    for name in SftWorkloadProfile.__dataclass_fields__:
        if name in provenance:
            continue
        assert name in base._content(), f"{name} is outside the content digest"
    assert set(base._content()) | provenance == set(SftWorkloadProfile.__dataclass_fields__)


def test_sample_policy_must_describe_the_rows_the_profile_measured() -> None:
    from flash.workload_profile import sft_sample_policy

    assert sft_sample_policy(0) == "exact-full"
    assert sft_sample_policy(None) == "exact-full"
    assert sft_sample_policy(64) == "exact-prefix"
    # a full-dataset profile that selected fewer rows than the dataset holds is describing a
    # prefix, and a quote reading "every row" off it would be quoting the wrong workload.
    with pytest.raises(ValueError, match="every source example"):
        _profile(sample_policy="exact-full")
    with pytest.raises(ValueError, match="sample_policy"):
        _profile(sample_policy="sampled")


def test_created_at_rejects_values_that_are_not_a_timestamp() -> None:
    for bad in (-1.0, float("nan"), float("inf"), True, "1780000000"):
        with pytest.raises(ValueError, match="created_at"):
            _profile(created_at=bad)


def test_sft_profile_key_changes_for_every_workload_shaping_input() -> None:
    spec = _spec()
    base = sft_profile_input_digest(
        spec,
        tokenizer_revision="tokenizer-a",
        producer_version="1.2.3",
    )
    variants = [
        replace(spec, seed=43),
        replace(spec, thinking=False),
        replace(spec, model_revision="e" * 40),
        replace(spec, environment=replace(spec.environment, resolved_sha="f" * 40)),
        replace(spec, environment=replace(spec.environment, params={"dataset_split": "eval"})),
        replace(spec, train=replace(spec.train, epochs=3)),
        replace(spec, train=replace(spec.train, batch_size=16)),
        replace(spec, train=replace(spec.train, max_context_tokens=2048)),
        replace(spec, train=replace(spec.train, max_steps=13)),
        replace(spec, train=replace(spec.train, max_examples=63)),
        replace(spec, worker_env={"FLASH_FLA_SM120": "pydelta"}),
    ]

    for variant in variants:
        assert (
            sft_profile_input_digest(
                variant,
                tokenizer_revision="tokenizer-a",
                producer_version="1.2.3",
            )
            != base
        )
    assert (
        sft_profile_input_digest(
            spec,
            tokenizer_revision="tokenizer-b",
            producer_version="1.2.3",
        )
        != base
    )
    assert (
        sft_profile_input_digest(
            spec,
            tokenizer_revision="tokenizer-a",
            producer_version="1.2.4",
        )
        != base
    )


def test_sft_profile_input_payload_hashes_environment_params_and_worker_env() -> None:
    payload = sft_profile_input_payload(
        _spec(),
        tokenizer_revision="tokenizer-a",
        producer_version="1.2.3",
    )
    encoded = json.dumps(payload, sort_keys=True)

    assert "private_selector" not in encoded
    assert "not-persisted" not in encoded
    assert "FLASH_FLA_SM120" not in encoded
    assert "params_sha256" in encoded
    assert "worker_env_sha256" in encoded


def test_sft_profile_contains_only_aggregate_evidence() -> None:
    encoded = json.dumps(_profile().to_dict(), sort_keys=True)

    for forbidden in ("prompt", "completion", "input_ids", "token_ids", "credential", "secret"):
        assert forbidden not in encoded.lower()


def test_sft_profile_run_id_is_deterministic_and_safe() -> None:
    digest = "a" * 64

    assert sft_profile_run_id(digest) == f"profile-sft-{digest}"
    with pytest.raises(ValueError, match="sha256"):
        sft_profile_run_id("not-a-digest")


def test_profile_carrier_is_internal_and_round_trips_only_in_worker_specs() -> None:
    profile = _profile().to_dict()
    spec = replace(
        _spec(),
        workload_profile_kind="sft",
        workload_profile_input_digest="c" * 64,
        workload_profile_producer_version="1.2.3",
        workload_profile=profile,
    )

    public = spec.to_dict()
    internal = spec.to_internal_dict()

    assert "workload_profile_kind" not in public
    assert "workload_profile_input_digest" not in public
    assert "workload_profile_producer_version" not in public
    assert "workload_profile" not in public
    assert JobSpec.from_dict(internal) == spec
