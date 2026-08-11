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
    from flash.core.catalog import MODELS

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
    from flash.core.catalog import MODELS, ModelInfo
    from flash.core.spec import JobSpec

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


def test_worker_image_disk_floor_needs_an_image_override(monkeypatch):
    """The floor describes the OVERRIDE's footprint, so it is inert on the published image."""
    from flash.providers._lifecycle.worker import worker_image_disk_floor

    monkeypatch.setenv("FLASH_WORKER_IMAGE_DISK_GB", "300")
    monkeypatch.delenv("FLASH_WORKER_IMAGE", raising=False)
    assert worker_image_disk_floor() == 0  # no override: must not resize ordinary runs

    monkeypatch.setenv("FLASH_WORKER_IMAGE", "ghcr.io/example/fat-worker:cu128")
    assert worker_image_disk_floor() == 300


def test_worker_image_disk_floor_refuses_a_malformed_value(monkeypatch):
    """A typo must not silently read as "no floor" -- that reinstates the disk failure it prevents."""
    import pytest

    from flash.providers._lifecycle.worker import worker_image_disk_floor

    monkeypatch.setenv("FLASH_WORKER_IMAGE", "ghcr.io/example/fat-worker:cu128")
    for bad in ("200GB", "", "-1", "1.5"):
        monkeypatch.setenv("FLASH_WORKER_IMAGE_DISK_GB", bad)
        if bad == "":
            assert worker_image_disk_floor() == 0  # unset is the only benign empty case
            continue
        with pytest.raises(ValueError, match="FLASH_WORKER_IMAGE_DISK_GB"):
            worker_image_disk_floor()

    # the override gate is checked BEFORE the value is parsed, so a stale typo left in the
    # environment cannot fail runs on the published image -- the floor is not theirs to obey.
    monkeypatch.delenv("FLASH_WORKER_IMAGE", raising=False)
    monkeypatch.setenv("FLASH_WORKER_IMAGE_DISK_GB", "200GB")
    assert worker_image_disk_floor() == 0


def test_submit_raises_disk_to_the_worker_image_floor(monkeypatch):
    """A custom worker image larger than the catalog sizing gets a lever (raise-only).

    ``[gpu] disk_gb`` is platform-managed (MANAGED_GPU_KEYS), so no user can author around an
    oversized override; without this the job fails during pull or extraction with no configuration
    lever at all.
    """
    from flash import runner
    from flash.core.catalog import MODELS, ModelInfo
    from flash.core.spec import JobSpec

    monkeypatch.setitem(
        MODELS,
        "test/small-disk",
        ModelInfo(
            id="test/small-disk",
            display_name="x",
            params="4B",
            params_b=4.0,
            algos=("sft",),
            min_vram_gb=32,
            min_disk_gb=100,
        ),
    )
    monkeypatch.setenv("FLASH_WORKER_IMAGE", "ghcr.io/example/fat-worker:cu128")
    monkeypatch.setenv("FLASH_WORKER_IMAGE_DISK_GB", "300")
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(tmp, "runs"))
        spec = JobSpec.from_dict(
            {
                "model": "test/small-disk",
                "algorithm": "sft",
                "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
                "train": {"max_examples": 8},
                "gpu": {"type": "RTX 5090", "disk_gb": 60},
            }
        )
        satisfy_sft_profile(runner, monkeypatch, spec)
        status = runner.submit_job(spec, dry_run=True)
        # image floor (300) beats the catalog floor (100) and the spec value (60)
        assert status.effective_preparation["worker_spec"]["gpu"]["disk_gb"] == 300

        # raise-only: a model needing MORE than the image keeps its own larger floor
        monkeypatch.setenv("FLASH_WORKER_IMAGE_DISK_GB", "80")
        status = runner.submit_job(spec, dry_run=True)
        assert status.effective_preparation["worker_spec"]["gpu"]["disk_gb"] == 100


def test_no_provider_is_disqualified_by_the_image_disk_floor(monkeypatch):
    """The floor sizes disk; it must not change which substrates are eligible.

    Lambda cannot honor ANY disk floor -- ``live_candidates`` never reads ``constraints.disk_gb``
    and says so -- but that is true of the catalog floor too, and a 200 GB catalog model already
    allocates there today. Refusing Lambda for the image floor alone would single out the smaller
    of two identical requests, dropping real capacity and failing a Lambda-pinned run to prevent
    nothing the platform prevents already.

    A guard here also cannot be written correctly: ``constraints.disk_gb`` arrives POST-max (see
    ``seed_submission``, which passes the already-raised ``attempt_spec.gpu.disk_gb``), so the
    floor can never exceed it and any such comparison is dead code. Honoring disk on Lambda is a
    pre-existing gap for the catalog floor and belongs with it, not on this knob.
    """
    from flash.providers.lambda_ import LambdaProvider
    from flash.providers.runpod import RunpodProvider
    from flash.providers.vast import VastProvider

    monkeypatch.setenv("FLASH_WORKER_IMAGE", "ghcr.io/example/fat-worker:cu128")
    monkeypatch.delenv("FLASH_WORKER_IMAGE_REGISTRY_AUTH", raising=False)
    monkeypatch.setenv("FLASH_WORKER_IMAGE_DISK_GB", "300")

    # no provider gained a disk-floor eligibility gate: the knob sizes disk, it does not veto
    for provider in (LambdaProvider(), VastProvider(), RunpodProvider()):
        assert not hasattr(provider, "_image_disk_floor_problem"), provider.name


def test_profile_job_carries_the_worker_image_disk_floor(monkeypatch):
    """The cpu profile job pulls the same override image, so it needs the same floor.

    It drops the model's disk sizing (nothing downloads weights) and never reaches
    ``_with_model_disk``, so an oversized image would fail the profile and block the sft
    submission gated behind it.
    """
    from flash import runner
    from flash.core.spec import JobSpec

    monkeypatch.setenv("FLASH_WORKER_IMAGE", "ghcr.io/example/fat-worker:cu128")
    monkeypatch.setenv("FLASH_WORKER_IMAGE_DISK_GB", "300")
    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "sft",
            "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
            "train": {"max_examples": 8},
            "gpu": {"type": "RTX 5090"},
        }
    )
    prepared = runner._prepared_sft_profile_job(spec, input_digest="d" * 64)
    assert prepared.worker_spec.gpu.disk_gb == 300

    # unset, the profile keeps the platform default it has always used
    monkeypatch.delenv("FLASH_WORKER_IMAGE_DISK_GB", raising=False)
    prepared = runner._prepared_sft_profile_job(spec, input_digest="d" * 64)
    assert prepared.worker_spec.gpu.disk_gb == runner.GpuSpec.disk_gb
