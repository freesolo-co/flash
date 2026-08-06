"""CPU tests for worker container-disk sizing (gpu.disk_gb wiring)."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

from tests._helpers.profile import satisfy_sft_profile


@dataclass
class _FakeTemplate:
    containerDiskInGb: int | None = 64


@dataclass
class _FakeConfig:
    template: _FakeTemplate | None = field(default_factory=_FakeTemplate)


def test_apply_disk_gb_raises_disk():
    from flash.providers.runpod.jobs import apply_disk_gb

    cfg = _FakeConfig()
    apply_disk_gb(cfg, 160)
    assert cfg.template.containerDiskInGb == 160


def test_apply_disk_gb_never_shrinks():
    """Configs carry a historical disk_gb=60 default; it must not shrink the 64 default."""
    from flash.providers.runpod.jobs import apply_disk_gb

    cfg = _FakeConfig()
    apply_disk_gb(cfg, 60)
    assert cfg.template.containerDiskInGb == 64


def test_apply_disk_gb_noops():
    from flash.providers.runpod.jobs import apply_disk_gb

    cfg = _FakeConfig()
    apply_disk_gb(cfg, None)
    apply_disk_gb(cfg, 0)
    assert cfg.template.containerDiskInGb == 64
    apply_disk_gb(_FakeConfig(template=None), 160)  # missing template: warn, don't raise


def test_oversized_catalog_models_carry_disk_floors():
    """Models whose transient download peak exceeds the shared cache need per-job disk floors."""
    from flash.catalog import MODELS

    disk_floor_models = {
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.6-27B",
        "Qwen/Qwen3.6-35B-A3B",
    }
    assert all(m.min_disk_gb == 0 for m in MODELS.values() if m.id not in disk_floor_models)
    assert MODELS["Qwen/Qwen3.5-9B"].min_disk_gb == 120
    assert MODELS["Qwen/Qwen3.6-27B"].min_disk_gb == 160
    assert MODELS["Qwen/Qwen3.6-35B-A3B"].min_disk_gb == 200


def test_submit_raises_disk_to_model_min(monkeypatch):
    """submit_job (dry-run) bumps gpu.disk_gb to a catalog model's min_disk_gb."""
    from flash import runner
    from flash.catalog import MODELS, ModelInfo
    from flash.spec import JobSpec

    monkeypatch.setitem(
        MODELS,
        "test/big-disk",
        ModelInfo(
            id="test/big-disk",
            display_name="x",
            params="4B",
            params_b=4.0,
            algos=("sft",),
            min_vram_gb=32,
            min_disk_gb=160,
        ),
    )
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(tmp, "runs"))
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
        # resolve. Disk sizing is what is under test, so seed the profile instead.
        satisfy_sft_profile(runner, monkeypatch, spec)
        status = runner.submit_job(spec, dry_run=True)
        # disk_gb is platform-managed: stripped from the public status.spec, read the sizing the
        # worker executes from the effective-preparation worker spec.
        assert status.effective_preparation["worker_spec"]["gpu"]["disk_gb"] == 160
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
        status = runner.submit_job(spec_big, dry_run=True)
        assert status.effective_preparation["worker_spec"]["gpu"]["disk_gb"] == 200
