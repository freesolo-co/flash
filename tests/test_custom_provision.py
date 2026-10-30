"""Standalone workers receive model credentials only through provider secrets."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    ("explicit", "cached"),
    [(None, None), (None, "cached-token"), ("explicit-token", "cached-token")],
)
def test_standalone_hf_credentials_are_forwarded_only_as_secrets(monkeypatch, explicit, cached):
    import huggingface_hub
    import modal

    from flash.custom.client import TrainingClient

    sandbox = Mock(object_id="sandbox-test")
    sandbox.tunnels.return_value = {8000: SimpleNamespace(url="https://gpu.example")}
    create = Mock(return_value=sandbox)
    secret = Mock(return_value="secret-object")
    image = Mock()
    image.pip_install.return_value = image
    image.add_local_dir.return_value = image
    monkeypatch.setattr(modal.Image, "from_registry", lambda *a: image)
    monkeypatch.setattr(modal.App, "lookup", lambda *a, **k: "app")
    monkeypatch.setattr(modal.Sandbox, "create", create)
    monkeypatch.setattr(modal.Secret, "from_dict", secret)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: cached)
    fake = SimpleNamespace(_http=SimpleNamespace(timeout=10))
    monkeypatch.setattr("flash.custom.client.TrainingClient", lambda *a, **k: fake)

    result = TrainingClient.create("private/model", hf_token=explicit)
    secret_values = secret.call_args.args[0]
    assert secret_values.get("HF_TOKEN") == (explicit or cached)
    assert set(secret_values) == (
        {"FLASH_CUSTOM_TOKEN", "HF_TOKEN"} if explicit or cached else {"FLASH_CUSTOM_TOKEN"}
    )
    assert create.call_args.kwargs["secrets"] == ["secret-object"]
    visible = json.dumps(create.call_args.kwargs["env"])
    for token in (explicit, cached):
        if token:
            assert token not in visible
    result._cleanup()
    sandbox.terminate.assert_called_once()
