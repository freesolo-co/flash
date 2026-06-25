"""The dev-channel (`freesolo-flash-dev` / `flash-dev`) derivation and build transforms.

The dev channel is the same source as `freesolo-flash`, built with a single flipped line in
flash/_channel.py (CHANNEL "prod" -> "dev"). These tests pin both ends of that contract: the
checked-in source is the prod channel, and the build-time rewrites produce a coherent dev package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_build_module():
    path = REPO_ROOT / "scripts" / "build_dev_dist.py"
    spec = importlib.util.spec_from_file_location("build_dev_dist", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checked_in_source_is_prod_channel():
    from flash import _channel
    from flash.client import config

    assert _channel.CHANNEL == "prod"
    assert _channel.CLI_NAME == "flash"
    assert _channel.DIST_NAME == "freesolo-flash"
    # The functional default everything reads from.
    assert config.DEFAULT_API_URL == "https://flash.freesolo.co"


def test_default_api_url_follows_channel():
    from flash.client.config import DEV_API_URL, PROD_API_URL, default_api_url

    assert default_api_url("prod") == PROD_API_URL == "https://flash.freesolo.co"
    assert default_api_url("dev") == DEV_API_URL == "https://flash-dev.freesolo.co"


def test_dev_channel_marker_derives_dev_names():
    # Run the real flash/_channel.py through the build rewrite and confirm everything that differs
    # between channels (CLI name, dist name, and thus the default URL) flips together.
    build = _load_build_module()
    channel_src = (REPO_ROOT / "flash" / "_channel.py").read_text()
    dev_src = build.rewrite_channel(channel_src)

    namespace: dict = {}
    exec(compile(dev_src, "flash/_channel.py[dev]", "exec"), namespace)
    assert namespace["CHANNEL"] == "dev"
    assert namespace["CLI_NAME"] == "flash-dev"
    assert namespace["DIST_NAME"] == "freesolo-flash-dev"


def test_rewrite_channel_requires_exactly_one_prod_marker():
    import pytest

    build = _load_build_module()
    with pytest.raises(SystemExit):
        build.rewrite_channel('CHANNEL = "dev"\n')  # already flipped -> zero matches


def test_rewrite_pyproject_retargets_only_the_project_table():
    build = _load_build_module()
    src = (
        "[project]\n"
        'name = "freesolo-flash"\n'
        'version = "0.2.25"\n'
        "dependencies = []\n"
        "\n"
        "[project.scripts]\n"
        'flash = "flash.cli.main:main"\n'
        "# Operator-only console script.\n"
        'flash-server = "flash.server.__main__:main"\n'
        "\n"
        "[tool.flash-dev]\n"
        'version = "9.9.9"\n'
    )
    out = build.rewrite_pyproject(src, "9.9.9")

    # Package + scripts renamed for a side-by-side dev install.
    assert 'name = "freesolo-flash-dev"' in out
    assert 'flash-dev = "flash.cli.main:main"' in out
    assert 'flash-dev-server = "flash.server.__main__:main"' in out
    # The console-script *values* (import targets) are untouched.
    assert '"flash.cli.main:main"' in out
    # The prod version is replaced by the dev version, and the identically-valued
    # [tool.flash-dev].version line is left alone (so both now read 9.9.9 -> two occurrences).
    assert '"0.2.25"' not in out
    assert out.count('version = "9.9.9"') == 2
    # The dist-name retarget did not bleed into the [tool.flash-dev] table header.
    assert "[tool.flash-dev]" in out


def test_read_dev_version():
    build = _load_build_module()
    src = '[project]\nname = "freesolo-flash"\nversion = "1.0.0"\n[tool.flash-dev]\nversion = "2.3.4"\n'
    assert build.read_dev_version(src) == "2.3.4"


def test_channels_are_at_the_same_version():
    # The prod (freesolo-flash) and dev (freesolo-flash-dev) channels must ship the same version;
    # .github/workflows/version-parity.yml enforces this in CI too.
    import tomllib

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert data["project"]["version"] == data["tool"]["flash-dev"]["version"]
