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
        args = types.SimpleNamespace(api_key="fs-secret-123", api_url=None, freesolo_url=None)
        rc = cli.cmd_login(args)
        assert rc == 0
        assert verified["api_key"] == "fs-secret-123"  # the key was actually verified

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
