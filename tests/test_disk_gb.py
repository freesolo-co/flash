"""CPU tests for worker container-disk sizing (gpu.disk_gb wiring)."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field


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


def test_default_catalog_models_need_no_disk_bump():
    """The dense Qwen3.5 catalog fits the platform's default container disk (min_disk_gb == 0).

    The only exception is the Qwen3.6 35B-A3B MoE: its ~70 GB bf16 checkpoint + HF/Xet temp
    overflow the 64 GB default, so it carries an explicit disk floor.
    """
    from flash.catalog import MODELS

    assert all(
        m.min_disk_gb == 0 for m in MODELS.values() if m.id != "Qwen/Qwen3.6-35B-A3B"
    )
    assert MODELS["Qwen/Qwen3.6-35B-A3B"].min_disk_gb == 200


def test_submit_raises_disk_to_model_min(monkeypatch):
    """submit_job (dry-run) bumps gpu.disk_gb to a catalog model's min_disk_gb. No shipping model
    needs a bump now, so inject a synthetic big-checkpoint entry to exercise the mechanism."""
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
                "gpu": {"type": "RTX 5090", "disk_gb": 60},
            }
        )
        status = runner.submit_job(spec, dry_run=True)
        assert status.spec["gpu"]["disk_gb"] == 160
        # explicit larger user value wins
        spec_big = JobSpec.from_dict(
            {
                "model": "test/big-disk",
                "algorithm": "sft",
                "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
                "gpu": {"type": "RTX 5090", "disk_gb": 200},
            }
        )
        status = runner.submit_job(spec_big, dry_run=True)
        assert status.spec["gpu"]["disk_gb"] == 200
