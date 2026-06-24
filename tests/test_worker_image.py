from flash.providers.runpod.train.deps import WORKER_IMAGE, worker_image_for_gpu


def _clear_image_env(monkeypatch):
    for key in (
        "FLASH_WORKER_IMAGE",
        "FLASH_WORKER_IMAGE_TEMPLATE",
        "FLASH_WORKER_IMAGE_PER_SM",
    ):
        monkeypatch.delenv(key, raising=False)


def test_worker_image_defaults_to_base_for_durable_jobs(monkeypatch):
    _clear_image_env(monkeypatch)

    assert worker_image_for_gpu("RTX 4090") == WORKER_IMAGE


def test_worker_image_can_return_none_for_live_endpoint_default(monkeypatch):
    _clear_image_env(monkeypatch)

    assert worker_image_for_gpu("RTX 4090", allow_default=False) is None


def test_worker_image_absolute_override_wins(monkeypatch):
    _clear_image_env(monkeypatch)
    monkeypatch.setenv("FLASH_WORKER_IMAGE", "ghcr.io/freesolo-co/flash-worker:hotfix")
    monkeypatch.setenv("FLASH_WORKER_IMAGE_PER_SM", "1")

    assert worker_image_for_gpu("RTX 4090") == "ghcr.io/freesolo-co/flash-worker:hotfix"


def test_worker_image_per_sm_appends_arch_to_default_tag(monkeypatch):
    _clear_image_env(monkeypatch)
    monkeypatch.setenv("FLASH_WORKER_IMAGE_PER_SM", "1")

    assert worker_image_for_gpu("RTX 4090") == "ghcr.io/freesolo-co/flash-worker:cu128-sm89"
    assert worker_image_for_gpu("A100 SXM") == "ghcr.io/freesolo-co/flash-worker:cu128-sm80"


def test_worker_image_template_can_define_arch_tags(monkeypatch):
    _clear_image_env(monkeypatch)
    monkeypatch.setenv(
        "FLASH_WORKER_IMAGE_TEMPLATE",
        "ghcr.io/freesolo-co/flash-worker:cu128-{sm}-{gpu_short}",
    )

    assert worker_image_for_gpu("H100") == "ghcr.io/freesolo-co/flash-worker:cu128-sm90-h100"
