"""The dev-channel (`freesolo-flash-dev` / `flash-dev`) derivation and build transforms.

The dev channel is the same source as `freesolo-flash`, built with a single flipped line in
flash/_internal/channel.py (CHANNEL "prod" -> "dev"). These tests pin both ends of that contract: the
checked-in source is the prod channel, and the build-time rewrites produce a coherent dev package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_build_module():
    path = REPO_ROOT / "scripts" / "build_dev_dist.py"
    spec = importlib.util.spec_from_file_location("build_dev_dist", path)
    assert spec is not None, f"could not load a spec for {path}"
    assert spec.loader is not None, f"spec for {path} has no loader"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checked_in_source_is_prod_channel():
    from flash._internal import channel as _channel
    from flash.client import config

    assert _channel.CHANNEL == "prod"
    assert _channel.CLI_NAME == "flash"
    assert _channel.DIST_NAME == "freesolo-flash"
    # The functional defaults everything reads from.
    assert config.DEFAULT_API_URL == "https://flash.freesolo.co"

    from flash.serve import deploy

    assert deploy.DEFAULT_FREESOLO_SERVING_URL == "https://serve.freesolo.co"


def test_default_api_url_follows_channel():
    from flash.client.config import DEV_API_URL, PROD_API_URL, default_api_url

    assert default_api_url("prod") == PROD_API_URL == "https://flash.freesolo.co"
    assert default_api_url("dev") == DEV_API_URL == "https://flash-dev.freesolo.co"


def test_default_serving_url_follows_channel():
    """The serving plane is per-channel, exactly like the control plane above.

    Each channel's serving app is backed by its own Supabase project, so an org
    row exists in one of them and not the other. A dev-channel client defaulting to prod serving
    posts a dev `org_id` into the prod database and takes a 23503 foreign-key violation -- which
    reads as a serving outage but is a routing defect, and which no retry can clear.
    """
    from flash.serve.deploy import (
        DEV_FREESOLO_SERVING_URL,
        PROD_FREESOLO_SERVING_URL,
        default_serving_url,
    )

    assert default_serving_url("prod") == PROD_FREESOLO_SERVING_URL == "https://serve.freesolo.co"
    assert default_serving_url("dev") == DEV_FREESOLO_SERVING_URL == "https://serve-dev.freesolo.co"
    # The two planes must not collapse onto one host: that is the whole defect being fixed, and an
    # equality here would make every assertion above pass while routing dev traffic to prod.
    assert PROD_FREESOLO_SERVING_URL != DEV_FREESOLO_SERVING_URL


def test_every_hosted_default_flips_with_the_channel():
    """Build the dev channel for real and assert NO hosted default is left pointing at prod.

    This is the regression that was missing. `serve.freesolo.co` sat hardcoded in
    `flash/serve/deploy.py` while `client/config.py` derived its URL from CHANNEL, so the dev
    package shipped a prod serving endpoint and every dev deploy failed the prod org_id FK.
    Each half was internally consistent, so nothing was red -- the same shape as the earlier
    catalog drift between the flash and serving rank limits.

    Asserting per-constant would just re-encode the mistake, so this scans the built dev tree for
    any surviving prod host. It fails on the NEXT hosted endpoint added without a channel split,
    not only on the one already fixed.
    """
    import re

    build = _load_build_module()
    dev_channel_src = build.rewrite_channel(
        (REPO_ROOT / "flash" / "_internal" / "channel.py").read_text()
    )
    namespace: dict = {}
    exec(compile(dev_channel_src, "flash/_internal/channel.py[dev]", "exec"), namespace)
    dev_channel = namespace["CHANNEL"]
    assert dev_channel == "dev"

    from flash.client.config import default_api_url
    from flash.serve.deploy import default_serving_url

    # Every channel-derived default, evaluated as the built dev package would.
    dev_defaults = {
        "control plane (flash.client.config.default_api_url)": default_api_url(dev_channel),
        "serving plane (flash.serve.deploy.default_serving_url)": default_serving_url(dev_channel),
    }
    assert dev_defaults, "no hosted defaults collected -- the scan would pass vacuously"

    # A prod host is any freesolo.co host without a dev marker in its leftmost label.
    prod_host = re.compile(r"^https://(?!.*-dev\.)(?!.*dev\.)[a-z0-9-]+\.freesolo\.co$")
    leaked = {name: url for name, url in dev_defaults.items() if prod_host.match(url)}
    assert not leaked, (
        "dev-channel default(s) still address production: "
        + "; ".join(f"{name} -> {url}" for name, url in sorted(leaked.items()))
        + ". Each hosted plane is backed by its own Supabase project, so a dev client hitting a "
        "prod endpoint fails the org_id foreign key. Give the constant a prod/dev pair that "
        "derives from CHANNEL, as flash.client.config and flash.serve.deploy both do."
    )


def test_dev_channel_marker_derives_dev_names():
    # Run the real flash/_internal/channel.py through the build rewrite and confirm everything that differs
    # between channels (CLI name, dist name, and thus the default URL) flips together.
    build = _load_build_module()
    channel_src = (REPO_ROOT / "flash" / "_internal" / "channel.py").read_text()
    dev_src = build.rewrite_channel(channel_src)

    namespace: dict = {}
    exec(compile(dev_src, "flash/_internal/channel.py[dev]", "exec"), namespace)
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


def test_no_build_flips_channel_and_pyproject_together(tmp_path):
    """--no-build (publish-image.yml, before the `:dev` Docker build) must apply the FULL
    transform, not just the channel flip. flash/__init__.py resolves __version__ via
    importlib.metadata.version(DIST_NAME), and DIST_NAME derives from CHANNEL -- so a
    channel-only flip would point that lookup at a distribution name nothing installed,
    silently downgrading __version__ to "0+unknown" in the built image."""
    build = _load_build_module()
    (tmp_path / "flash" / "_internal").mkdir(parents=True)
    channel_path = tmp_path / "flash" / "_internal" / "channel.py"
    channel_path.write_text('CHANNEL = "prod"\n')
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        '[project]\nname = "freesolo-flash"\nversion = "1.0.0"\n'
        '[tool.flash-dev]\nversion = "1.0.0"\n'
    )

    rc = build.main(["--root", str(tmp_path), "--no-build"])

    assert rc == 0
    assert channel_path.read_text() == 'CHANNEL = "dev"\n'
    # The installed package name must move in lockstep with CHANNEL/DIST_NAME.
    assert 'name = "freesolo-flash-dev"' in pyproject_path.read_text()


def test_channels_are_at_the_same_version():
    # The prod (freesolo-flash) and dev (freesolo-flash-dev) channels must ship the same version;
    # .github/workflows/version-parity.yml enforces this in CI too.
    import tomllib

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert data["project"]["version"] == data["tool"]["flash-dev"]["version"]
