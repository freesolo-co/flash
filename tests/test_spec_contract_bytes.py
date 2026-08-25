"""Pin the serialized BYTES of the four job-spec contracts.

`JobSpec` is read through four different contracts -- authored config, public representation,
persisted recovery record, and resolved worker payload -- and `_preparation_digest`
(flash/runner/lifecycle/preparation.py) sha256-hashes the canonical JSON of two of them. That makes their
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
    MANAGED_ENVIRONMENT_KEYS,
    MANAGED_GPU_KEYS,
    MANAGED_SECTION_KEYS,
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
        "model": "Qwen/Qwen3.5-9B",
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
    "model_revision_force_pin",
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
MANAGED_ENVIRONMENT_FIELDS = {"package", "resolved_sha"}


def test_public_payload_emits_exactly_the_authorable_keys():
    assert set(spec().to_dict()) == PUBLIC_TOP_LEVEL


def test_worker_payload_is_the_public_key_set_plus_the_managed_fields():
    worker = set(spec().to_internal_dict())
    assert worker == PUBLIC_TOP_LEVEL | WORKER_ONLY_TOP_LEVEL
    # the public half must be a strict subset: a key that exists only publicly would be a field the
    # worker cannot see, which is how a resolved value silently reverts to a default on recovery.
    assert worker > PUBLIC_TOP_LEVEL


def test_public_payload_strips_every_managed_gpu_key():
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
                "model": "Qwen/Qwen3.5-9B",
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
                "model": "Qwen/Qwen3.5-9B",
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
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "sft",
                "project": PROJECT,
                "environment": {"id": "owner/project/env"},
                "train": {"epochs": 1},
                "not_a_real_field": 1,
            }
        )


def test_sft_batch_size_is_not_migrated():
    """`batch_size` means a different quantity on sft, so the rollout migration must not touch it."""
    decoded = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
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
            "model": "Qwen/Qwen3.5-9B",
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


def test_registries_match_the_declared_boundary():
    """The registries must equal this module's own literals, and nothing may leak the other way.

    Deliberately NOT `set(worker) - set(public) == MANAGED_TOP_LEVEL_KEYS`: that compares the
    registry against itself, so dropping a name shrinks both sides at once and the equation still
    balances while the field leaks publicly. Mutation-verified -- that exact sabotage survived the
    registry form. The payload difference itself is already pinned by
    `test_public_payload_emits_exactly_the_authorable_keys` and
    `test_worker_payload_is_the_public_key_set_plus_the_managed_fields`, so this only has to bind
    the registries to the same independent literals.
    """
    assert MANAGED_TOP_LEVEL_KEYS == WORKER_ONLY_TOP_LEVEL
    assert MANAGED_TRAIN_KEYS == MANAGED_TRAIN_FIELDS
    assert MANAGED_GPU_KEYS == MANAGED_GPU_FIELDS
    assert MANAGED_ENVIRONMENT_KEYS == MANAGED_ENVIRONMENT_FIELDS
    public, worker = spec().to_dict(), spec().to_internal_dict()
    # nothing travels the other way: the public payload invents no key the worker half lacks. this
    # is what catches an empty `providers` or `type_fallbacks` reaching the public bytes, which
    # would change every stored digest.
    assert not set(public) - set(worker)
    # every section present in the public payload, not just the registered ones. driving this loop
    # from the managed section registry would let a key in an unregistered section (`wandb`) reach
    # the public bytes unseen -- the same blind spot, in the opposite direction, as the section walk
    # in test_every_privately_held_field_is_named_in_a_registry.
    for section, public_section in public.items():
        if not isinstance(public_section, dict):
            continue
        assert not set(public_section) - set(worker.get(section) or {}), (
            f"the public payload emitted a [{section}] key the worker payload does not have"
        )


def test_every_privately_held_field_is_named_in_a_registry():
    """Nothing may be stripped from the public payload without being named in a registry.

    Both serializers are BLACKLISTS -- they emit everything and pop what is managed -- so a newly
    added field is PUBLIC by default here and was public by default before the projection too.
    Measured, not assumed: an unregistered field injected into the payload leaks into dev's public
    output and into this one identically. The projection does not change that direction.

    What it does change is that the boundary is now ENUMERABLE. A run of `data.pop(...)` statements
    cannot be asserted against; the registries can. Walks EVERY section, not just the top level:
    `environment.resolved_sha` was stripped inline and named in no registry, and a top-level-only
    version of this test passed while that was true.

    The sections walked are the ones actually PRESENT in the payload, not the ones named in
    `MANAGED_SECTION_KEYS`. Deriving them from the registry would make this test blind in exactly
    the direction it exists to cover: an inline strip inside an unregistered section (`wandb`) is
    the same defect as the `environment` one, and a registry-driven walk cannot see it.

    A section the public payload drops WHOLE is skipped: `workload_profile` is named in
    `MANAGED_TOP_LEVEL_KEYS`, so descending into it would re-report a strip the top-level check has
    already accounted for. The profile below is populated on purpose -- with an empty one that
    distinction costs nothing and the test passes either way.
    """
    populated_profile = {"packing_mode": "packed", "examples_per_update": 2, "packed_blocks": 1}
    candidate = spec(
        train={"epochs": 1, "init_from_adapter": "src/step-4"},
        workload_profile=populated_profile,
    )
    public, worker = candidate.to_dict(), candidate.to_internal_dict()
    # `project` is public-only by construction (it has no worker counterpart), and the two warm-start
    # topology keys are stripped conditionally, so no registry can express them.
    exempt = {"project", "lora_rank", "lora_alpha"}
    unregistered = {
        f"(top){name}" for name in set(worker) - set(public) - MANAGED_TOP_LEVEL_KEYS - exempt
    }
    registered = dict(MANAGED_SECTION_KEYS)
    for section, worker_section in worker.items():
        if not isinstance(worker_section, dict) or section not in public:
            continue
        public_section = public.get(section) or {}
        managed_keys = registered.get(section, frozenset())
        unregistered |= {
            f"{section}.{name}"
            for name in set(worker_section) - set(public_section) - set(managed_keys) - exempt
        }
    # `gpu.type_fallbacks` is a reshape, not a removal: it is folded into the public `gpu.type`.
    unregistered -= {"gpu.type_fallbacks"}
    assert not unregistered, (
        f"{sorted(unregistered)} are stripped from the public payload but named in no registry, so "
        "the public contract is defined by statements again rather than by the registries"
    )
