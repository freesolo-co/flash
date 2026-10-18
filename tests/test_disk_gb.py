"""CPU tests for worker container-disk sizing (gpu.disk_gb wiring)."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from math import ceil

import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.submit as runner_submit
from tests._helpers.profile import satisfy_sft_profile


@dataclass
class _FakeTemplate:
    containerDiskInGb: int | None = 64


@dataclass
class _FakeConfig:
    template: _FakeTemplate | None = field(default_factory=_FakeTemplate)


def test_apply_disk_gb_raises_disk():
    from flash.providers.runpod.execution.job_execution import apply_disk_gb

    cfg = _FakeConfig()
    apply_disk_gb(cfg, 160)
    assert cfg.template.containerDiskInGb == 160


def test_apply_disk_gb_never_shrinks():
    """Configs carry a historical disk_gb=60 default; it must not shrink the 64 default."""
    from flash.providers.runpod.execution.job_execution import apply_disk_gb

    cfg = _FakeConfig()
    apply_disk_gb(cfg, 60)
    assert cfg.template.containerDiskInGb == 64


def test_apply_disk_gb_noops():
    from flash.providers.runpod.execution.job_execution import apply_disk_gb

    cfg = _FakeConfig()
    apply_disk_gb(cfg, None)
    apply_disk_gb(cfg, 0)
    assert cfg.template.containerDiskInGb == 64
    apply_disk_gb(_FakeConfig(template=None), 160)  # missing template: warn, don't raise


def test_oversized_catalog_models_carry_disk_floors():
    """Models whose transient download peak exceeds the shared cache need per-job disk floors."""
    from flash.core.catalog import MODELS

    disk_floor_models = {
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.8-27B",
        "Qwen/Qwen3.6-35B-A3B",
    }
    assert all(m.min_disk_gb == 0 for m in MODELS.values() if m.id not in disk_floor_models)
    assert MODELS["Qwen/Qwen3.5-9B"].min_disk_gb == 120
    assert MODELS["Qwen/Qwen3.8-27B"].min_disk_gb == 160
    assert MODELS["Qwen/Qwen3.6-35B-A3B"].min_disk_gb == 200


def test_every_catalog_algorithm_gets_the_full_bf16_merge_floor():
    from flash.core.catalog import MODELS, resolve_model

    for model in MODELS.values():
        expected = max(model.min_disk_gb, ceil(model.params_b * 2) * 3 + 64)
        for algorithm in model.algos:
            assert resolve_model(model.id, algorithm).min_disk_gb == expected


def test_merge_floor_covers_the_concurrent_publish_peak():
    """The floor must fit the three full model copies a publish holds at once.

    Publishing overlaps training, so the container disk carries the checkpoint being merged, the
    next checkpoint training is writing, and the merger's full-model output -- and verl saves the
    whole `state_dict`, not the lora delta. Budgeting one copy is what let a 35B run train both of
    its steps and then die at publish with 24.98 GB free on a 200 GB disk.

    Asserted as an inequality against the modelled peak rather than by restating the formula: a
    test that recomputes `ceil(p*2)*3+64` passes for any multiplier, including the one that failed.
    """
    from flash.core.catalog import MODELS, resolve_model

    for model in MODELS.values():
        for algorithm in model.algos:
            resolved = resolve_model(model.id, algorithm)
            one_copy_gb = ceil(resolved.params_b * 2)
            # published checkpoint + concurrently-written next checkpoint + merged output.
            assert resolved.min_disk_gb >= one_copy_gb * 3


def test_public_model_rows_report_the_derived_merge_floor():
    from flash.core.catalog import MODELS, public_model_rows

    rows = {row["id"]: row for row in public_model_rows()}
    for model in MODELS.values():
        expected = max(model.min_disk_gb, ceil(model.params_b * 2) * 3 + 64)
        assert rows[model.id]["min_disk_gb"] == expected


def test_fractional_parameter_merge_floor_rounds_up(monkeypatch):
    from flash.core.catalog import MODELS, ModelInfo, resolve_model

    model = ModelInfo(
        id="test/fractional-disk",
        display_name="fractional",
        params="0.9B",
        params_b=0.9,
        algos=("sft",),
        min_vram_gb=1,
    )
    monkeypatch.setitem(MODELS, model.id, model)

    # ceil(0.9 * 2) == 2 rounds up before it is tripled, not after.
    assert resolve_model(model.id, "sft").min_disk_gb == 70


def test_moe_merge_floor_uses_total_parameters(monkeypatch):
    from flash.core.catalog import MODELS, ModelInfo, resolve_model

    model = ModelInfo(
        id="test/moe-disk",
        display_name="moe",
        params="35B total / 3B active",
        params_b=35.0,
        active_params_b=3.0,
        algos=("sft", "grpo", "opd"),
        min_vram_gb=1,
    )
    monkeypatch.setitem(MODELS, model.id, model)

    # sized on total parameters: the checkpoint and the merged output are dense full-model copies,
    # so the active-parameter count does not bound what lands on disk.
    assert resolve_model(model.id, "opd").min_disk_gb == 274


def test_revision_geometry_is_applied_before_the_disk_floor(monkeypatch):
    from flash.core.catalog import MODELS, ModelInfo, resolve_model
    from flash.engine.plan import vram

    model = ModelInfo(
        id="test/revision-disk",
        display_name="revision",
        params="4B",
        params_b=4.0,
        algos=("sft",),
        min_vram_gb=1,
    )
    monkeypatch.setitem(MODELS, model.id, model)
    monkeypatch.setattr(
        vram,
        "_validated_revision_geometry",
        lambda model_id, revision, info: (50.0, 123456),
    )

    resolved = resolve_model(model.id, "sft", "commit")
    assert resolved.params_b == 50.0
    assert resolved.min_disk_gb == 364


def test_submit_applies_derived_model_disk_floor(monkeypatch):
    """submit_job sends the resolved full-bf16 merge floor to the worker."""
    from flash.core.catalog import MODELS, ModelInfo
    from flash.core.spec import JobSpec

    model = ModelInfo(
        id="test/big-disk",
        display_name="x",
        params="8B",
        params_b=8.0,
        algos=("sft",),
        min_vram_gb=32,
        min_disk_gb=0,
    )
    monkeypatch.setitem(MODELS, model.id, model)
    expected_floor = ceil(model.params_b * 2) * 3 + 64
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner_state, "RUNS_DIR", os.path.join(tmp, "runs"))
        spec = JobSpec.from_dict(
            {
                "model": "test/big-disk",
                "algorithm": "sft",
                "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
                "train": {"max_examples": 8},
                "gpu": {"type": "RTX 5090", "disk_gb": 60},
            }
        )
        # sft submission is profile-gated, and this synthetic catalog model has no hub revision to
        # resolve. disk sizing is what is under test, so seed the profile instead.
        satisfy_sft_profile(monkeypatch, spec)
        status = runner_submit.submit_job(spec, dry_run=True)
        # disk_gb is platform-managed: stripped from the public status.spec, read the sizing the
        # worker executes from the effective-preparation worker spec.
        assert status.effective_preparation["worker_spec"]["gpu"]["disk_gb"] == expected_floor
        # explicit larger user value wins
        spec_big = JobSpec.from_dict(
            {
                "model": "test/big-disk",
                "algorithm": "sft",
                "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
                "train": {"max_examples": 8},
                "gpu": {"type": "RTX 5090", "disk_gb": 200},
            }
        )
        status = runner_submit.submit_job(spec_big, dry_run=True)
        assert status.effective_preparation["worker_spec"]["gpu"]["disk_gb"] == 200
