"""CPU tests for worker container-disk sizing (gpu.disk_gb wiring)."""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass, field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@dataclass
class _FakeTemplate:
    containerDiskInGb: int | None = 64


@dataclass
class _FakeConfig:
    template: _FakeTemplate | None = field(default_factory=_FakeTemplate)


def test_apply_disk_gb_raises_disk():
    from autoslm.flash.durable import apply_disk_gb

    cfg = _FakeConfig()
    apply_disk_gb(cfg, 160)
    assert cfg.template.containerDiskInGb == 160


def test_apply_disk_gb_never_shrinks():
    """Configs carry a historical disk_gb=60 default; it must not shrink the 64 default."""
    from autoslm.flash.durable import apply_disk_gb

    cfg = _FakeConfig()
    apply_disk_gb(cfg, 60)
    assert cfg.template.containerDiskInGb == 64


def test_apply_disk_gb_noops():
    from autoslm.flash.durable import apply_disk_gb

    cfg = _FakeConfig()
    apply_disk_gb(cfg, None)
    apply_disk_gb(cfg, 0)
    assert cfg.template.containerDiskInGb == 64
    apply_disk_gb(_FakeConfig(template=None), 160)  # missing template: warn, don't raise


def test_catalog_min_disk_for_big_moe():
    from autoslm.catalog import get_model

    assert get_model("Qwen/Qwen3.6-35B-A3B").min_disk_gb >= 100  # ~72 GB checkpoint
    assert get_model("Qwen/Qwen3-30B-A3B").min_disk_gb >= 100  # ~61 GB checkpoint
    assert get_model("Qwen/Qwen3-4B-Instruct-2507").min_disk_gb == 0


def test_submit_raises_disk_to_model_min(monkeypatch):
    """submit_job (dry-run) bumps gpu.disk_gb to the catalog's min_disk_gb."""
    from autoslm import orchestrator
    from autoslm.worker_spec import JobSpec

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(orchestrator, "RUNS_DIR", os.path.join(tmp, "runs"))
        spec = JobSpec.from_dict(
            {
                "model": "Qwen/Qwen3.6-35B-A3B",
                "algorithm": "sft",
                "environment": {"id": "gsm8k"},
                "gpu": {"type": "RTX 5090", "disk_gb": 60},
            }
        )
        status = orchestrator.submit_job(spec, dry_run=True)
        assert status.spec["gpu"]["disk_gb"] == 160
        # explicit larger user value wins
        spec_big = JobSpec.from_dict(
            {
                "model": "Qwen/Qwen3.6-35B-A3B",
                "algorithm": "sft",
                "environment": {"id": "gsm8k"},
                "gpu": {"type": "RTX 5090", "disk_gb": 200},
            }
        )
        status = orchestrator.submit_job(spec_big, dry_run=True)
        assert status.spec["gpu"]["disk_gb"] == 200
