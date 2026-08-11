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


def test_lambda_offers_nothing_while_a_disk_floor_it_cannot_honor_is_set(monkeypatch):
    """Lambda must not rent a paid box it is certain cannot extract the image.

    The floor reaches ``AllocationConstraints.disk_gb``, but Lambda sells fixed instance storage:
    its capacity probe ignores the disk constraint and its launch path has no disk-sizing field. An
    operator only sets the floor because the image does NOT fit the default sizing, so allocating
    here buys an instance that fails during extraction -- and a never-started worker is retriable,
    so the paid attempt could repeat. RunPod (endpoint template) and Vast (offer search) can both
    satisfy the floor and stay eligible.
    """
    import pytest

    from flash.providers.base import AllocationConstraints, UnsupportedGpuError
    from flash.providers.lambda_ import LambdaProvider
    from flash.providers.vast import VastProvider

    monkeypatch.setenv("FLASH_WORKER_IMAGE", "ghcr.io/example/fat-worker:cu128")
    monkeypatch.delenv("FLASH_WORKER_IMAGE_REGISTRY_AUTH", raising=False)
    monkeypatch.setenv("FLASH_WORKER_IMAGE_DISK_GB", "300")

    with pytest.raises(UnsupportedGpuError, match="FLASH_WORKER_IMAGE_DISK_GB"):
        LambdaProvider().live_candidates(24, AllocationConstraints())
    # vast searches offers on disk (_effective_disk_gb), so the floor must NOT disqualify it --
    # the guard is specific to the substrate that cannot size disk, not to the floor existing.
    assert not hasattr(VastProvider(), "_image_disk_floor_problem")

    # and with no floor set, lambda is eligible again
    monkeypatch.delenv("FLASH_WORKER_IMAGE_DISK_GB", raising=False)
    assert LambdaProvider()._image_disk_floor_problem() == ""


def test_lambda_stays_eligible_for_a_floor_that_raises_nothing(monkeypatch):
    """The refusal keys on the floor RAISING disk, not on it being set.

    ``_with_model_disk`` applies the floor as a raise-only maximum, so a value at or below the
    platform default asks for no more disk than an ordinary run already gets. Refusing those would
    drop real Lambda capacity on an automatic allocation, and fail a Lambda-pinned run as
    unsupported, while preventing nothing.
    """
    from flash.core.spec import GpuSpec
    from flash.providers.lambda_ import LambdaProvider

    monkeypatch.setenv("FLASH_WORKER_IMAGE", "ghcr.io/example/worker:cu128")

    for benign in (1, GpuSpec.disk_gb - 1, GpuSpec.disk_gb):
        monkeypatch.setenv("FLASH_WORKER_IMAGE_DISK_GB", str(benign))
        assert LambdaProvider()._image_disk_floor_problem() == "", benign

    # one GB above the default is the first value that actually asks for more disk
    monkeypatch.setenv("FLASH_WORKER_IMAGE_DISK_GB", str(GpuSpec.disk_gb + 1))
    assert "FLASH_WORKER_IMAGE_DISK_GB" in LambdaProvider()._image_disk_floor_problem()


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
