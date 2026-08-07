"""The platform-managed config fields are rejected as user input and stripped from the public spec.

These fields are assigned by the control plane / runner, never authored by the user: the top-level
``run_id``, the per-run artifact repo ``train.hf_repo``, the runner-sized
``gpu.disk_gb`` and the weight-cache ``gpu.network_volume`` / ``gpu.network_volume_gb``, the
control-plane-pinned ``environment.resolved_sha``, and the retry/wall-clock lifecycle policy
``gpu.max_retries`` / ``gpu.max_wall_seconds``. ``train.lora_alpha`` is managed too but derived, not
merely assigned: it is always ``2 * lora_rank``. The user-facing parser (``spec_from_dict``) rejects
each loudly instead of silently dropping it, and ``JobSpec.to_dict()`` (the public, user-authorable
representation) omits them all so the client-submit -> server-revalidation round trip never trips
over a managed value. ``JobSpec.to_internal_dict()`` (the control-plane/worker carrier) retains them.
"""

from __future__ import annotations

import pytest

from flash.schema import ConfigError, spec_from_dict
from flash.spec import JobSpec
from tests._helpers.specs import raw_spec

# (id, raw-dict override applied to a minimal valid spec, expected error fragment). Each override
# adds exactly one managed field as user input; the parser must reject on that field.
_MANAGED_REJECTIONS = [
    ("run_id", {"run_id": "user-chosen"}, r"unknown config key\(s\): run_id"),
    (
        "train.hf_repo",
        {"train": {"epochs": 1, "max_examples": 8, "hf_repo": "me/runs"}},
        r"\[train\] unknown key\(s\): hf_repo",
    ),
    (
        "train.lora_alpha",
        {"train": {"epochs": 1, "max_examples": 8, "lora_alpha": 128}},
        r"\[train\] unknown key\(s\): lora_alpha",
    ),
    ("gpu.disk_gb", {"gpu": {"disk_gb": 999}}, r"\[gpu\] unknown key\(s\): disk_gb"),
    (
        "gpu.network_volume",
        {"gpu": {"network_volume": "my-vol"}},
        r"\[gpu\] unknown key\(s\): network_volume",
    ),
    (
        "gpu.network_volume_gb",
        {"gpu": {"network_volume_gb": 500}},
        r"\[gpu\] unknown key\(s\): network_volume_gb",
    ),
    ("gpu.max_retries", {"gpu": {"max_retries": 9}}, r"\[gpu\] unknown key\(s\): max_retries"),
    (
        "gpu.max_wall_seconds",
        {"gpu": {"max_wall_seconds": 7200}},
        r"\[gpu\] unknown key\(s\): max_wall_seconds",
    ),
    (
        "environment.resolved_sha",
        {"environment": {"id": "owner/env", "resolved_sha": "a" * 40}},
        r"\[environment\] unknown key\(s\): resolved_sha",
    ),
]


@pytest.mark.parametrize(
    ("override", "match"),
    [(o, m) for _id, o, m in _MANAGED_REJECTIONS],
    ids=[_id for _id, _o, _m in _MANAGED_REJECTIONS],
)
def test_managed_field_rejected_as_user_input(override, match):
    # A user who sets a platform-managed field in their config is rejected loudly, not silently
    # ignored, so a mistaken value can never be mistaken for an honored one.
    with pytest.raises(ConfigError, match=match):
        spec_from_dict(raw_spec(**override))


def _fully_managed_internal_spec() -> JobSpec:
    """An internal spec carrying every managed field (the worker's view)."""
    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "sft",
            "environment": {
                "id": "github:owner/repo@main:env/environment.py",
                "resolved_sha": "b" * 40,
            },
            "train": {"epochs": 1, "max_examples": 8, "lora_rank": 16, "hf_repo": "operator/runs"},
            "gpu": {
                "type": "",
                "disk_gb": 160,
                "network_volume": "flash-weights",
                "network_volume_gb": 100,
                "max_retries": 9,
                "max_wall_seconds": 7200,
            },
            "run_id": "flash-managed-run",
            "model_policy": "catalog",
        }
    )
    # sanity: the internal carrier really does hold every managed value
    assert spec.run_id == "flash-managed-run"
    assert spec.model_policy == "catalog"
    assert spec.train.hf_repo == "operator/runs"
    assert spec.train.lora_alpha == 32  # derived: 2 * lora_rank (16)
    assert spec.gpu.disk_gb == 160
    assert spec.gpu.network_volume == "flash-weights"
    assert spec.gpu.network_volume_gb == 100
    assert spec.gpu.max_retries == 9
    assert spec.gpu.max_wall_seconds == 7200
    assert spec.environment.resolved_sha == "b" * 40
    return spec


def test_public_spec_omits_all_managed_fields():
    public = _fully_managed_internal_spec().to_dict()
    assert "run_id" not in public
    assert "model_policy" not in public
    assert "hf_repo" not in public["train"]
    assert "lora_alpha" not in public["train"]
    assert not (
        {"disk_gb", "network_volume", "network_volume_gb", "max_retries", "max_wall_seconds"}
        & set(public["gpu"])
    )
    assert "resolved_sha" not in public["environment"]


def test_internal_dict_retains_all_managed_fields():
    internal = _fully_managed_internal_spec().to_internal_dict()
    assert internal["run_id"] == "flash-managed-run"
    assert internal["model_policy"] == "catalog"
    assert internal["train"]["hf_repo"] == "operator/runs"
    assert internal["train"]["lora_alpha"] == 32  # derived value travels to the worker
    assert internal["gpu"]["disk_gb"] == 160
    assert internal["gpu"]["network_volume"] == "flash-weights"
    assert internal["gpu"]["network_volume_gb"] == 100
    assert internal["gpu"]["max_retries"] == 9
    assert internal["gpu"]["max_wall_seconds"] == 7200
    assert internal["environment"]["resolved_sha"] == "b" * 40


def test_public_spec_round_trips_through_user_parser():
    # The client submits spec.to_dict(); the server re-validates it with the SAME user-facing parser.
    # Because to_dict() omits every managed field, that round trip must pass without a rejection -
    # the run_id travels as the separate server-assigned parameter, not in the config body.
    public = _fully_managed_internal_spec().to_dict()
    reparsed = spec_from_dict(public, run_id="flash-managed-run")
    assert reparsed.model == "Qwen/Qwen3.5-0.8B"
    assert reparsed.run_id == "flash-managed-run"  # from the parameter, not the config body
    # the server re-derives managed defaults; the user never supplied them
    assert reparsed.train.hf_repo == ""
    assert reparsed.gpu.network_volume is None
    assert reparsed.gpu.max_retries == 5  # GpuSpec default
    assert reparsed.gpu.max_wall_seconds == 24 * 3600  # GpuSpec default
    assert reparsed.environment.resolved_sha == ""
    # lora_rank survives the public round trip, and alpha is re-derived from it, not defaulted.
    assert reparsed.train.lora_rank == 16
    assert reparsed.train.lora_alpha == 32


@pytest.mark.parametrize(("lora_rank", "expected_alpha"), [(16, 32), (32, 64), (8, 16)])
def test_lora_alpha_derived_as_twice_rank(lora_rank, expected_alpha):
    # lora_alpha is not a user knob: the user picks lora_rank and alpha is always 2 x rank.
    spec = spec_from_dict(raw_spec(train={"epochs": 1, "max_examples": 8, "lora_rank": lora_rank}))
    assert spec.train.lora_alpha == expected_alpha
    assert "lora_alpha" not in spec.to_dict()["train"]


def test_default_lora_rank_derives_alpha_64():
    # With no lora_rank authored, rank defaults to 32 so the derived alpha is 64.
    spec = spec_from_dict(raw_spec())
    assert spec.train.lora_rank == 32
    assert spec.train.lora_alpha == 64


def test_internal_from_dict_round_trips_stored_lora_alpha():
    # The internal carrier preserves a stored alpha so a warm-start's inherited parent alpha (which
    # need not equal 2 x rank for a legacy source adapter) survives control-plane -> worker
    # serialization; alpha is re-derived only when absent (a stripped public to_dict being reparsed).
    base = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "grpo",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "train": {"epochs": 1, "max_examples": 8, "lora_rank": 16, "lora_alpha": 48},
    }
    assert JobSpec.from_dict(base).train.lora_alpha == 48  # present -> round-trip
    absent = {**base, "train": {"epochs": 1, "max_examples": 8, "lora_rank": 16}}
    assert JobSpec.from_dict(absent).train.lora_alpha == 32  # absent -> derive 2 x rank
