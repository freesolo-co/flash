from types import SimpleNamespace

import flash.providers._lifecycle.net.worker as worker_module
from flash.providers._lifecycle.net.worker import WORKER_IMAGE, worker_image_for_gpu


def test_worker_image_defaults_to_per_sm_baked_tag_for_durable_jobs():
    # The per-SM baked kernel-cache image is always used for baked arches (RTX 4090 = sm89),
    # so durable jobs get the warmed -smXX tag with no env knob required.
    assert worker_image_for_gpu("RTX 4090") == "ghcr.io/freesolo-co/flash-worker:cu128-sm89"


def test_worker_image_appends_arch_to_default_tag():
    assert worker_image_for_gpu("RTX 4090") == "ghcr.io/freesolo-co/flash-worker:cu128-sm89"
    assert worker_image_for_gpu("A100 SXM") == "ghcr.io/freesolo-co/flash-worker:cu128-sm80"


def test_worker_image_uses_sm100_baked_tag_for_b200():
    assert worker_image_for_gpu("B200") == "ghcr.io/freesolo-co/flash-worker:cu128-sm100"
    assert worker_image_for_gpu("RTX 5090") == "ghcr.io/freesolo-co/flash-worker:cu128-sm120"


def test_worker_image_falls_back_to_base_for_unbaked_arch(monkeypatch):
    monkeypatch.setattr(
        worker_module,
        "get_gpu_info",
        lambda _friendly_gpu: SimpleNamespace(sm="sm999"),
    )

    assert worker_image_for_gpu("future gpu") == WORKER_IMAGE


def test_worker_image_is_not_configurable_by_environment(monkeypatch):
    """No env var may redirect the worker image: every run uses the published tag for its arch.

    The FLASH_WORKER_IMAGE override was removed once it was established that nothing set it. This
    pins the absence, so reintroducing a reader is a test failure rather than a silent new knob.
    """
    for key in ("FLASH_WORKER_IMAGE", "FLASH_WORKER_IMAGE_TEMPLATE", "FLASH_WORKER_IMAGE_DISK_GB"):
        monkeypatch.setenv(key, "ghcr.io/attacker/not-ours:latest")

    assert worker_image_for_gpu("H100") == "ghcr.io/freesolo-co/flash-worker:cu128-sm90"
