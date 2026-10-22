from types import SimpleNamespace

import pytest

import flash.providers._lifecycle.net.worker as worker_module
from flash.providers._lifecycle.net.worker import WORKER_IMAGE, worker_image_for_gpu


@pytest.mark.parametrize(
    "gpu", [None, "L4", "L40S", "A100 SXM", "H100", "H200", "B200", "RTX Pro 6000"]
)
def test_worker_uses_qualified_upstream_image_on_every_architecture(gpu):
    # live qualification found the mutable cu128-sm90 tag still installed the retired fork.
    # lock the actual launch reference, including the fallback used by cpu publication.
    assert worker_image_for_gpu(gpu) == (
        "ghcr.io/freesolo-co/flash-worker@"
        "sha256:34c84092f87b9ae43457620a8e5c0dc2e3dc30fdd5de5b4300c1eeb384e9a066"
    )


def test_worker_image_uses_qualified_base_for_unbaked_arch(monkeypatch):
    monkeypatch.setattr(worker_module, "get_gpu_info", lambda _gpu: SimpleNamespace(sm="sm999"))
    assert worker_image_for_gpu("future gpu") == WORKER_IMAGE


def test_worker_image_is_not_configurable_by_environment(monkeypatch):
    for key in ("FLASH_WORKER_IMAGE", "FLASH_WORKER_IMAGE_TEMPLATE", "FLASH_WORKER_IMAGE_DISK_GB"):
        monkeypatch.setenv(key, "ghcr.io/attacker/not-ours:latest")
    assert worker_image_for_gpu("H100") == WORKER_IMAGE
