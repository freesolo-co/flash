"""Pin the serialized BYTES of the four job-spec contracts.

`JobSpec` is read through four different contracts -- authored config, public representation,
persisted recovery record, and resolved worker payload -- and `_preparation_digest`
(flash/runner/preparation.py) sha256-hashes the canonical JSON of two of them. That makes their
serialized bytes a recovery contract rather than an implementation detail: a refactor that changes
which KEYS `to_dict()` or `to_internal_dict()` emit invalidates the stored digest of every
warm-start and workload-profile run in flight, and those runs then fail integrity validation on
recovery instead of resuming.

Ordinary spec tests cannot catch that. They assert on parsed field VALUES, so they stay green while
the emitted key set moves underneath them. These assert on the bytes.

The record shapes below are taken from the persisted corpus in `~/.flash/runs` (1552 records at the
time of writing) so the legacy shapes exercised here are ones that actually exist on disk: 1529
records still carry `[worker_env]` and 536 still carry `model_policy`.

Run: uv run pytest tests/test_spec_contract_bytes.py -q
"""

from __future__ import annotations

import json

import pytest

from flash.core.spec import (
    MANAGED_GPU_KEYS,
    MANAGED_TOP_LEVEL_KEYS,
    MANAGED_TRAIN_KEYS,
    JobSpec,
)

PROJECT = "11111111-1111-4111-8111-111111111111"


def canonical(payload: dict) -> str:
    """Serialize the way `_preparation_digest` does, so these assertions bind the same bytes."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def spec(**overrides) -> JobSpec:
    raw = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "sft",
        "project": PROJECT,
        "environment": {"id": "owner/project/env"},
        "train": {"epochs": 1, "max_examples": 8},
        "gpu": {"type": "H100"},
    }
    raw.update(overrides)
    return JobSpec.from_dict(raw)


# the public half is what `status.spec` stores and what the authored parser must accept back. every
# key here is one an author can write; anything platform-managed leaking in would break the resubmit
# that recovery and `flash runs get` perform.
PUBLIC_TOP_LEVEL = {
    "algorithm",
    "environment",
    "gpu",
    "model",
    "project",
    "seed",
    "thinking",
    "train",
    "wandb",
}

# the worker half additionally carries every resolved and platform-managed field. these are exactly
# the keys `to_dict()` strips, and they are the reason a public spec cannot be handed to a worker.
WORKER_ONLY_TOP_LEVEL = {
    "gpu_count_auto",
    "model_revision",
    "model_revision_auto",
    "run_id",
    "workload_profile",
    "workload_profile_input_digest",
    "workload_profile_producer_version",
}

# the same boundary for the two nested sections. spelled out here rather than imported so the
# registries in `flash.core.spec` have something independent to be checked against.
MANAGED_TRAIN_FIELDS = {"hf_repo", "init_from_adapter_revision"}
MANAGED_GPU_FIELDS = {
    "disk_gb",
    "network_volume",
    "network_volume_gb",
    "max_retries",
    "max_wall_seconds",
}


def test_public_payload_emits_exactly_the_authorable_keys():
    assert set(spec().to_dict()) == PUBLIC_TOP_LEVEL


def test_worker_payload_is_the_public_key_set_plus_the_managed_fields():
    worker = set(spec().to_internal_dict())
    assert worker == PUBLIC_TOP_LEVEL | WORKER_ONLY_TOP_LEVEL
    # the public half must be a strict subset: a key that exists only publicly would be a field the
    # worker cannot see, which is how a resolved value silently reverts to a default on recovery.
    assert worker > PUBLIC_TOP_LEVEL


def test_public_payload_strips_every_managed_gpu_key():
    from flash.core.spec import MANAGED_GPU_KEYS

    public_gpu = set(spec().to_dict()["gpu"])
    assert public_gpu.isdisjoint(MANAGED_GPU_KEYS)
    assert set(spec().to_internal_dict()["gpu"]) >= MANAGED_GPU_KEYS


def test_public_payload_strips_control_plane_train_fields():
    public_train = set(spec().to_dict()["train"])
    assert "hf_repo" not in public_train
    assert "init_from_adapter_revision" not in public_train


def test_public_payload_strips_the_resolved_environment_pin():
    assert "resolved_sha" not in spec().to_dict()["environment"]


def test_environment_pip_travels_in_the_public_payload():
    """Author-declared scorer dependencies are public: only the author can declare them."""
    pinned = spec(environment={"id": "owner/project/env", "pip": ["requests==2.31.0"]})
    assert pinned.to_dict()["environment"]["pip"] == ("requests==2.31.0",)


# --- digest invariance -------------------------------------------------------------------------
# what the digest does and does not bind. these two tests are the guard rail: they state in one
# place that reordering keys is free but changing which keys exist is not.


def test_digest_bytes_ignore_key_order():
    payload = spec().to_dict()
    reordered = dict(reversed(list(payload.items())))
    assert canonical(reordered) == canonical(payload)


def test_digest_bytes_change_when_a_key_is_added_or_dropped():
    payload = spec().to_dict()
    dropped = {k: v for k, v in payload.items() if k != "seed"}
    assert canonical(dropped) != canonical(payload)
    added = {**payload, "unexpected": 1}
    assert canonical(added) != canonical(payload)


def test_internal_round_trip_is_byte_stable():
    """`from_dict(to_internal_dict())` is the recovery read; it must not drift on re-serialization."""
    original = spec()
    once = original.to_internal_dict()
    twice = JobSpec.from_dict(once).to_internal_dict()
    assert canonical(twice) == canonical(once)


def test_empty_collections_stay_omitted_rather_than_explicit():
    """An omitted empty field and an explicit empty one hash differently, so the pops are load-bearing."""
    payload = spec().to_internal_dict()
    assert "type_fallbacks" not in payload["gpu"]
    assert "providers" not in payload["gpu"]


# --- persisted-record tolerance ----------------------------------------------------------------
# the persisted decoder is deliberately more permissive than the authored parser. these pin the
# specific historical shapes that still exist on disk.


@pytest.mark.parametrize("dropped_key", ["model_policy", "worker_env", "workload_profile_kind"])
def test_persisted_records_still_decode_with_removed_top_level_keys(dropped_key):
    raw = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "sft",
        "project": PROJECT,
        "environment": {"id": "owner/project/env"},
        "train": {"epochs": 1},
        dropped_key: {"SOME_KEY": "value"} if dropped_key == "worker_env" else "allow",
    }
    decoded = JobSpec.from_dict(raw)
    # tolerated on READ only: the removed key must not reappear in either serialized contract.
    assert dropped_key not in decoded.to_dict()
    assert dropped_key not in decoded.to_internal_dict()


def test_persisted_records_still_decode_with_removed_train_keys():
    decoded = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "sft",
            "project": PROJECT,
            "environment": {"id": "owner/project/env"},
            "train": {"epochs": 1, "advantage_clip": 0.2},
        }
    )
    assert "advantage_clip" not in decoded.to_internal_dict()["train"]


def test_removed_train_key_tolerance_does_not_extend_to_other_sections():
    """The retired-key allowlist is per-section, not global.

    `validated_section` takes `removed` as an argument, so the allowlist is now something a caller
    passes rather than something the decoder hardcodes -- and passing it to the wrong section is a
    failure the inline code it replaced could not express. `advantage_clip` is a retired [train] key;
    under [gpu] it is just an unknown key and must stay fatal. Without this, widening the allowlist to
    another section would let a stored key vanish silently instead of failing.
    """
    with pytest.raises(ValueError, match="gpu has unknown key"):
        JobSpec.from_dict(
            {
                "model": "Qwen/Qwen3.5-0.8B",
                "algorithm": "sft",
                "project": PROJECT,
                "environment": {"id": "owner/project/env"},
                "gpu": {"type": "H100 SXM", "advantage_clip": 0.2},
            }
        )


@pytest.mark.parametrize(
    ("section", "payload"),
    [("train", {"epochs": 1, "nope": 1}), ("gpu", {"type": "H100 SXM", "nope": 1})],
)
def test_unknown_keys_inside_a_section_are_fatal(section, payload):
    """Unknown-key rejection has to hold INSIDE a section, not just at the top level.

    Both nested blocks are read through one shared validator now, so a regression there would open
    every section at once -- and a top-level-only assertion stays green while it happens.
    """
    with pytest.raises(ValueError, match=f"{section} has unknown key"):
        JobSpec.from_dict(
            {
                "model": "Qwen/Qwen3.5-0.8B",
                "algorithm": "sft",
                "project": PROJECT,
                "environment": {"id": "owner/project/env"},
                section: payload,
            }
        )


def test_unknown_persisted_keys_are_still_fatal():
    """Tolerance is an explicit allowlist, not a blanket ignore, or a typo silently becomes a default."""
    with pytest.raises(ValueError, match="unknown key"):
        JobSpec.from_dict(
            {
                "model": "Qwen/Qwen3.5-0.8B",
                "algorithm": "sft",
                "project": PROJECT,
                "environment": {"id": "owner/project/env"},
                "train": {"epochs": 1},
                "not_a_real_field": 1,
            }
        )


@pytest.mark.parametrize("algorithm", ["grpo", "opd"])
def test_pre_1_1_43_rollout_batch_migrates_to_prompts_per_step(algorithm):
    """Rollout runs written before the rename carry only `batch_size`; workers read the new name."""
    decoded = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": algorithm,
            "project": PROJECT,
            "environment": {"id": "owner/project/env"},
            "train": {"batch_size": 32, "group_size": 4},
        }
    )
    assert decoded.train.prompts_per_step == 32
    # MOVED, not copied: a spec carrying a MEANINGFUL value under both names is rejected by the
    # authored parser, which would break the resubmit that recovery performs. `to_dict()` still
    # emits the key (it emits every train field), so what matters is that its value is now None.
    assert decoded.train.batch_size is None
    assert decoded.to_dict()["train"]["batch_size"] is None


@pytest.mark.parametrize("algorithm", ["grpo", "opd"])
def test_new_rollout_spelling_wins_when_a_record_carries_both(algorithm):
    decoded = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": algorithm,
            "project": PROJECT,
            "environment": {"id": "owner/project/env"},
            "train": {"batch_size": 64, "prompts_per_step": 32, "group_size": 4},
        }
    )
    assert decoded.train.prompts_per_step == 32
    assert decoded.train.batch_size is None


def test_sft_batch_size_is_not_migrated():
    """`batch_size` means a different quantity on sft, so the rollout migration must not touch it."""
    decoded = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "sft",
            "project": PROJECT,
            "environment": {"id": "owner/project/env"},
            "train": {"batch_size": 8},
        }
    )
    assert decoded.train.batch_size == 8
    assert decoded.train.prompts_per_step is None


def test_ordered_gpu_pin_round_trips_through_the_public_spelling():
    """Public list form and internal head-plus-fallbacks form are alternate spellings of one pin."""
    decoded = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "sft",
            "project": PROJECT,
            "environment": {"id": "owner/project/env"},
            "train": {"epochs": 1},
            "gpu": {"type": ["H100", "A100"]},
        }
    )
    # note the alias canonicalization: "A100" is stored as its canonical name, and BOTH halves must
    # agree on it or the two spellings would hash differently for the same authored pin.
    assert decoded.to_dict()["gpu"]["type"] == ["H100", "A100 PCIe"]
    internal = decoded.to_internal_dict()["gpu"]
    assert internal["type"] == "H100"
    assert internal["type_fallbacks"] == ("A100 PCIe",)


def test_public_payload_is_the_worker_payload_minus_the_managed_registries():
    """The projection's contract, stated as an equation rather than as two parallel serializers.

    `to_dict()` is built from `to_internal_dict()`, so the only structural difference between the
    public and worker contracts is the three MANAGED_*_KEYS sets. If a future edit reintroduces a
    second `asdict(self)` and the two drift, this fails on the difference rather than at a recovery
    digest months later.
    """
    public, worker = spec().to_dict(), spec().to_internal_dict()
    # Compared against the literal key sets this module already declares, NOT against the registries
    # themselves. `set(worker) - set(public) == MANAGED_TOP_LEVEL_KEYS` reads well but compares the
    # registry against itself: dropping a name shrinks both sides at once, so the equation balances
    # while the field leaks publicly. Mutation-verified -- that exact sabotage survived the registry
    # form of this assertion.
    assert set(worker) - set(public) == WORKER_ONLY_TOP_LEVEL
    assert set(worker["train"]) - set(public["train"]) == MANAGED_TRAIN_FIELDS
    assert set(worker["gpu"]) - set(public["gpu"]) == MANAGED_GPU_FIELDS
    # and the registries must equal those same independent literals, so a drifting registry and a
    # drifting serializer cannot cancel out above.
    assert MANAGED_TOP_LEVEL_KEYS == WORKER_ONLY_TOP_LEVEL
    assert MANAGED_TRAIN_KEYS == MANAGED_TRAIN_FIELDS
    assert MANAGED_GPU_KEYS == MANAGED_GPU_FIELDS
    # nothing travels the other way: the public payload invents no key the worker half lacks.
    assert not set(public) - set(worker)
    assert not set(public["train"]) - set(worker["train"])
    assert not set(public["gpu"]) - set(worker["gpu"]), (
        "the public payload emitted a [gpu] key the worker payload does not have; an empty "
        "`providers` or `type_fallbacks` reaching the public bytes changes every stored digest"
    )


def test_a_new_private_field_must_be_registered_to_stay_private():
    """Public-by-default is the point of the projection, so prove the default actually holds.

    Under the old two-serializer shape a new field was PRIVATE by default: it reached
    `to_internal_dict` via `asdict` and stayed out of `to_dict` until someone remembered to add it,
    so forgetting was silent and the field never reached the public contract. Now forgetting is
    loud in the opposite direction -- the field appears publicly -- which is the failure a test can
    actually catch.
    """
    registered = MANAGED_TOP_LEVEL_KEYS | {"project"}
    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "sft",
            "project": PROJECT,
            "environment": {"id": "owner/project/env"},
            "train": {"epochs": 1},
            "gpu": {"type": "H100"},
        }
    )
    unregistered = {
        name for name in spec.to_internal_dict() if name not in spec.to_dict()
    } - registered
    assert not unregistered, (
        f"{sorted(unregistered)} are stripped from the public payload but named in no registry, so "
        "the public contract is defined by statements again rather than by MANAGED_*_KEYS"
    )
