from types import SimpleNamespace

import pytest

import flash.providers._lifecycle.net.worker as worker_module
from flash.providers._lifecycle.net.worker import (
    QUALIFIED_WORKER_IMAGES,
    WORKER_IMAGE,
    worker_image_for_gpu,
)


@pytest.mark.parametrize(
    ("gpu", "sm"),
    [
        ("A100 SXM", "sm80"),
        ("A100 SXM 40GB", "sm80"),
        ("A10", "sm86"),
        ("L4", "sm89"),
        ("L40S", "sm89"),
        ("H100", "sm90"),
        ("H200", "sm90"),
        ("B200", "sm100"),
        ("RTX Pro 6000", "sm120"),
    ],
)
def test_worker_selects_qualified_cache_image_for_gpu_architecture(gpu, sm):
    assert worker_image_for_gpu(gpu) == QUALIFIED_WORKER_IMAGES[sm]
    assert worker_image_for_gpu(gpu) != WORKER_IMAGE


@pytest.mark.parametrize("gpu", [None, ""])
def test_worker_without_gpu_keeps_qualified_base(gpu):
    assert (
        worker_image_for_gpu(gpu)
        == WORKER_IMAGE
        == (
            "ghcr.io/freesolo-co/flash-worker@"
            "sha256:34c84092f87b9ae43457620a8e5c0dc2e3dc30fdd5de5b4300c1eeb384e9a066"
        )
    )


def test_bake_inventory_alone_does_not_enable_an_image(monkeypatch):
    monkeypatch.delitem(QUALIFIED_WORKER_IMAGES, "sm90")
    assert "sm90" in worker_module.BAKED_PER_SM_ARCHES
    assert worker_image_for_gpu("H100") == WORKER_IMAGE


def test_qualified_images_are_immutable_and_distinct():
    import re

    assert set(QUALIFIED_WORKER_IMAGES) <= worker_module.BAKED_PER_SM_ARCHES
    assert len(set(QUALIFIED_WORKER_IMAGES.values())) == len(QUALIFIED_WORKER_IMAGES)
    for image in QUALIFIED_WORKER_IMAGES.values():
        assert re.fullmatch(r"ghcr\.io/freesolo-co/flash-worker@sha256:[0-9a-f]{64}", image)


def test_worker_image_uses_qualified_base_for_unbaked_arch(monkeypatch):
    monkeypatch.setattr(worker_module, "get_gpu_info", lambda _gpu: SimpleNamespace(sm="sm999"))
    assert worker_image_for_gpu("future gpu") == WORKER_IMAGE


def test_worker_image_is_not_configurable_by_environment(monkeypatch):
    for key in ("FLASH_WORKER_IMAGE", "FLASH_WORKER_IMAGE_TEMPLATE", "FLASH_WORKER_IMAGE_DISK_GB"):
        monkeypatch.setenv(key, "ghcr.io/attacker/not-ours:latest")
    assert worker_image_for_gpu("H100") == QUALIFIED_WORKER_IMAGES["sm90"]


def test_worker_image_rejects_unknown_gpu():
    with pytest.raises(ValueError, match="unsupported gpu"):
        worker_image_for_gpu("not a GPU")
