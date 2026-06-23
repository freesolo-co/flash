"""`flash login` verifies the freesolo key, then stores it with private (0600) permissions."""

from __future__ import annotations

import importlib
import json
import os
import stat
import tempfile
import types


def test_login_writes_private_config(monkeypatch):
    saved_home = os.environ.get("HOME")
    try:
        _check_login(monkeypatch)
    finally:
        if saved_home is not None:
            os.environ["HOME"] = saved_home
        import flash.client.config as client_config

        importlib.reload(client_config)


def _check_login(monkeypatch):
    with tempfile.TemporaryDirectory() as home:
        os.environ["HOME"] = home
        import flash.client.config as client_config

        importlib.reload(client_config)
        import flash.cli.main as cli

        # Login verifies the freesolo key against the freesolo backend, then stores it. Stub
        # the network verify so the test stays offline; an invalid key would raise instead.
        verified: dict = {}
        monkeypatch.setattr(
            cli.commands,
            "verify_freesolo_key",
            lambda api_key, base_url=None: verified.update(api_key=api_key, base_url=base_url),
        )
        # After storing the key, login fetches the identity card straight from the key it just
        # verified (not ambient credentials); stub the client so the test stays offline and
        # capture the key it was built with.
        identity = {"kind": "freesolo_api_key", "key_prefix": "fs-secr", "email": "me@example.com"}
        built_with: dict = {}

        class _FakeApi:
            def __init__(self, api_url, api_key=None, timeout=None):
                built_with.update(api_url=api_url, api_key=api_key)

            def me(self):
                return identity

        monkeypatch.setattr(cli.commands, "ApiClient", _FakeApi)
        args = types.SimpleNamespace(api_key="fs-secret-123", api_url=None, freesolo_url=None)
        rc = cli.cmd_login(args)
        assert rc == 0
        assert verified["api_key"] == "fs-secret-123"  # the key was actually verified
        assert built_with["api_key"] == "fs-secret-123"  # ...and that exact key built the card

        cfg = client_config.CONFIG_PATH
        assert cfg.exists()
        mode = stat.S_IMODE(os.stat(cfg).st_mode)
        assert mode == 0o600, f"expected 0600, got {oct(mode)}"
        # Directory should be private too.
        dir_mode = stat.S_IMODE(os.stat(client_config.CONFIG_DIR).st_mode)
        assert dir_mode == 0o700, f"expected 0700, got {oct(dir_mode)}"
        assert json.loads(cfg.read_text())["api_key"] == "fs-secret-123"
        # And the stored key resolves through the normal credential lookup.
        os.environ.pop("FREESOLO_API_KEY", None)
        _, key = client_config.load_credentials()
        assert key == "fs-secret-123"


def test_login_warns_when_env_key_will_override_saved_key(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("FREESOLO_API_KEY", "fs-secret-env")

    import flash.client.config as client_config

    importlib.reload(client_config)
    import flash.cli.main as cli

    monkeypatch.setattr(cli.commands, "verify_freesolo_key", lambda api_key, base_url=None: None)
    args = types.SimpleNamespace(api_key="fs-secret-arg", api_url=None, freesolo_url=None)

    assert cli.cmd_login(args) == 0
    captured = capsys.readouterr()
    assert "logged in to flash" in captured.out
    assert "FREESOLO_API_KEY is set and will override this saved login" in captured.err
    assert client_config.load_credentials()[1] == "fs-secret-env"
