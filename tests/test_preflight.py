import pytest

from flash.providers import preflight


@pytest.mark.parametrize(
    "key", ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "HF_TOKEN", "FREESOLO_INTERNAL_KEY"]
)
def test_missing_required_configuration(monkeypatch, key):
    monkeypatch.setenv("HF_TOKEN", "test-hf")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "test-internal")
    monkeypatch.delenv(key)
    with pytest.raises(preflight.PreflightError):
        preflight.require_operator_config()


def test_modal_configuration_is_sufficient(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-hf")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "test-internal")
    preflight.require_operator_config()


def test_standalone_requires_artifact_namespace(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-hf")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "test-internal")
    monkeypatch.setenv("FLASH_STANDALONE", "1")
    with pytest.raises(preflight.PreflightError, match="FLASH_HF_NAMESPACE"):
        preflight.require_operator_config()
