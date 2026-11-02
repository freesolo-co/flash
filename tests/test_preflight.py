import pytest

from flash.providers.core import preflight


@pytest.mark.parametrize(
    "key",
    [
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "FLASH_ARTIFACT_BLOB_ACCOUNT_URL",
        "FREESOLO_INTERNAL_KEY",
    ],
)
def test_missing_required_configuration(monkeypatch, key):
    monkeypatch.setenv(
        "FLASH_ARTIFACT_BLOB_ACCOUNT_URL", "https://testartifacts.blob.core.windows.net"
    )
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "test-internal")
    monkeypatch.delenv(key)
    with pytest.raises(preflight.PreflightError):
        preflight.require_operator_config()


def test_modal_configuration_needs_no_hub_write_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv(
        "FLASH_ARTIFACT_BLOB_ACCOUNT_URL", "https://testartifacts.blob.core.windows.net"
    )
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "test-internal")
    preflight.require_operator_config()


def test_standalone_uses_operator_azure_storage(monkeypatch):
    monkeypatch.setenv(
        "FLASH_ARTIFACT_BLOB_ACCOUNT_URL", "https://testartifacts.blob.core.windows.net"
    )
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "test-internal")
    monkeypatch.setenv("FLASH_STANDALONE", "1")
    preflight.require_operator_config()
