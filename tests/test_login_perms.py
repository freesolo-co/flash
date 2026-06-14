"""`slm login` stores the AutoSLM API key with private (0600) permissions."""

from __future__ import annotations

import importlib
import json
import os
import stat
import sys
import tempfile
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_login_writes_private_config():
    saved_home = os.environ.get("HOME")
    try:
        _check_login()
    finally:
        if saved_home is not None:
            os.environ["HOME"] = saved_home
        import autoslm.client.config as client_config

        importlib.reload(client_config)


def _check_login():
    with tempfile.TemporaryDirectory() as home:
        os.environ["HOME"] = home
        import autoslm.client.config as client_config

        importlib.reload(client_config)
        import autoslm.cli.main as cli

        # `--api-key` stores an existing key without contacting the control plane.
        args = types.SimpleNamespace(
            api_key="sk-autoslm-secret-123", email=None, api_url=None, debug=False
        )
        rc = cli.cmd_login(args)
        assert rc == 0

        cfg = client_config.CONFIG_PATH
        assert cfg.exists()
        mode = stat.S_IMODE(os.stat(cfg).st_mode)
        assert mode == 0o600, f"expected 0600, got {oct(mode)}"
        # Directory should be private too.
        dir_mode = stat.S_IMODE(os.stat(client_config.CONFIG_DIR).st_mode)
        assert dir_mode == 0o700, f"expected 0700, got {oct(dir_mode)}"
        assert json.loads(cfg.read_text())["api_key"] == "sk-autoslm-secret-123"
        # And the stored key resolves through the normal credential lookup.
        os.environ.pop("AUTOSLM_API_KEY", None)
        _, key = client_config.load_credentials()
        assert key == "sk-autoslm-secret-123"
