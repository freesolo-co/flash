from flash.providers.base import get_gpu_info
from flash.providers.runpod.train.deps import (
    BAKED_PER_SM_ARCHES,
    WORKER_IMAGE,
    worker_image_for_gpu,
)


def _clear_image_env(monkeypatch):
    for key in (
        "FLASH_WORKER_IMAGE",
        "FLASH_WORKER_IMAGE_TEMPLATE",
    ):
        monkeypatch.delenv(key, raising=False)


def test_worker_image_defaults_to_per_sm_baked_tag_for_durable_jobs(monkeypatch):
    _clear_image_env(monkeypatch)

    # The per-SM baked kernel-cache image is now always used for baked arches (RTX 4090 = sm89),
    # so durable jobs get the warmed -smXX tag with no env knob required.
    assert worker_image_for_gpu("RTX 4090") == "ghcr.io/freesolo-co/flash-worker:cu128-sm89"


def test_worker_image_can_return_none_for_live_endpoint_default(monkeypatch):
    _clear_image_env(monkeypatch)

    assert worker_image_for_gpu("RTX 4090", allow_default=False) is None


def test_worker_image_absolute_override_wins(monkeypatch):
    _clear_image_env(monkeypatch)
    monkeypatch.setenv("FLASH_WORKER_IMAGE", "ghcr.io/freesolo-co/flash-worker:hotfix")

    assert worker_image_for_gpu("RTX 4090") == "ghcr.io/freesolo-co/flash-worker:hotfix"


def test_worker_image_appends_arch_to_default_tag(monkeypatch):
    _clear_image_env(monkeypatch)

    assert worker_image_for_gpu("RTX 4090") == "ghcr.io/freesolo-co/flash-worker:cu128-sm89"
    assert worker_image_for_gpu("A100 SXM") == "ghcr.io/freesolo-co/flash-worker:cu128-sm80"


def test_worker_image_falls_back_to_base_for_unbaked_arch(monkeypatch):
    # B200 is sm100, which has no per-SM kernel-cache image baked yet, so it must fall back to the
    # base WORKER_IMAGE (cold-JIT) rather than select a `cu128-sm100` tag that `docker pull` would
    # 404 on. A baked arch (sm120 / RTX 5090) still gets its suffix.
    _clear_image_env(monkeypatch)

    assert get_gpu_info("B200").sm == "sm100"
    assert "sm100" not in BAKED_PER_SM_ARCHES
    assert worker_image_for_gpu("B200") == WORKER_IMAGE  # no -sm100 tag selected
    # a published arch is unaffected
    assert worker_image_for_gpu("RTX 5090") == "ghcr.io/freesolo-co/flash-worker:cu128-sm120"


def test_worker_image_template_can_define_arch_tags(monkeypatch):
    _clear_image_env(monkeypatch)
    monkeypatch.setenv(
        "FLASH_WORKER_IMAGE_TEMPLATE",
        "ghcr.io/freesolo-co/flash-worker:cu128-{sm}-{gpu_short}",
    )

    assert worker_image_for_gpu("H100") == "ghcr.io/freesolo-co/flash-worker:cu128-sm90-h100"


def test_per_sm_and_template_never_leak_into_live_endpoints(monkeypatch):
    # per-sm and template images are durable runpod queue-worker images (cmd runs rp_handler.py); they
    # must not route into the live-endpoint path (allow_default=False), which expects none unless the
    # absolute flash_worker_image override names a live-compatible image.
    _clear_image_env(monkeypatch)
    monkeypatch.setenv(
        "FLASH_WORKER_IMAGE_TEMPLATE",
        "ghcr.io/freesolo-co/flash-worker:cu128-{sm}",
    )

    assert worker_image_for_gpu("RTX 4090", allow_default=False) is None
    # the absolute override is still honored on the live path
    monkeypatch.setenv("FLASH_WORKER_IMAGE", "ghcr.io/freesolo-co/flash-worker:live")
    assert worker_image_for_gpu("RTX 4090", allow_default=False) == (
        "ghcr.io/freesolo-co/flash-worker:live"
    )
